"""
PHASE 1: Inner-loop only (CPG + RevDE-style DE) on a small suite of viable bodies.

What it does:
- Randomly samples N_BODIES genotypes (3 NDE input vectors), decodes bodies.
- Viability filters: actuators present + gentle short probe progress.
- For each viable body, tunes a compact CPG (per-actuator amplitude/phase/bias + global freq/ramp)
  with a small-population DE (RevDE-style params F=0.5, CR=0.9).
- Fitness function is UNCHANGED (negative distance to TARGET_POSITION).
- Saves best robot JSON + a video of the learned controller for each body.

After this works reliably, we’ll plug this inner loop under the outer body EA.
"""

from pathlib import Path
from typing import Any, List, Tuple, Optional

import numpy as np
import numpy.typing as npt
import mujoco as mj

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer

# ----------------------------------------------------
# RNG / Paths
# ----------------------------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)

CWD = Path.cwd()
OUTPUT = CWD / "__phase1__"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Assignment / world constants
# ----------------------------------------------------
NUM_MODULES = 30
GENE_LEN = 64                         # length of each of the three NDE input vectors
SPAWN_POS = [-0.8, 0.0, 0.10]         # we'll add +0.08 z for stability
TARGET_POSITION = [5.0, 0.0, 0.5]     # used by the unchanged fitness function

# Simulation durations for inner-loop episodes (dynamic)
DUR_SHORT = 12.0
DUR_MED = 35.0
DUR_LONG = 80.0
# Checkpoints in forward progress (x) from spawn
CP1 = 1.5
CP2 = 4.5

# Probe (non-learner) filter (gentler & looser)
PROBE_DURATION = 3.0
NONLEARNER_EPS = 0.05                 # min forward progress required in probe

# Body suite size
N_BODIES = 4                          # small suite to validate learning across morphologies
MAX_ATTEMPTS = 600                    # allow more attempts to find viable bodies

# Controller filtering/smoothing
QUIET_TIME = 0.25                     # initial time with zero control to settle
SMOOTH_ALPHA = 0.9                    # exponential smoothing for controls (0=none, 0.9=strong smoothing)

# ----------------------------------------------------
# CPG parameterization (per actuator)
# params vector structure (length = 3*nu + 2):
#   [A_0..A_{nu-1}, phi_0..phi_{nu-1}, b_0..b_{nu-1}, freq, ramp]
# ----------------------------------------------------
def cpg_bounds(nu: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns lower and upper bounds for the CPG parameter vector.
    """
    # Safe amplitude: [0, pi/2]
    A_lo = np.zeros(nu, dtype=np.float64)
    A_hi = (np.pi / 2.0) * np.ones(nu, dtype=np.float64)
    # Phase: [0, 2pi]
    PH_lo = np.zeros(nu, dtype=np.float64)
    PH_hi = (2.0 * np.pi) * np.ones(nu, dtype=np.float64)
    # Bias: [-pi/6, pi/6]
    BIAS_MAX = np.pi / 6.0
    B_lo = -BIAS_MAX * np.ones(nu, dtype=np.float64)
    B_hi = +BIAS_MAX * np.ones(nu, dtype=np.float64)
    # Global frequency: [0.3, 1.5] Hz
    f_lo, f_hi = 0.3, 1.5
    # Ramp constant: [0.1, 0.8] s^-1
    r_lo, r_hi = 0.1, 0.8

    lo = np.concatenate([A_lo, PH_lo, B_lo, np.array([f_lo, r_lo])])
    hi = np.concatenate([A_hi, PH_hi, B_hi, np.array([f_hi, r_hi])])
    return lo, hi


def cpg_init(nu: int, rng: np.random.Generator) -> np.ndarray:
    """
    Initializes CPG params with helpful priors:
    - Small amplitudes to start.
    - Phases clustered to even≈0, odd≈pi (plus noise).
    - Bias near zero.
    - Mid frequency and ramp.
    """
    lo, hi = cpg_bounds(nu)
    params = np.empty_like(lo)

    # Amplitudes: small initial magnitudes
    A0 = 0.2 * (np.pi / 2.0)
    params[:nu] = rng.uniform(0.0, A0, size=nu)

    # Phases: two clusters for alternation
    phases = np.where((np.arange(nu) % 2) == 0, 0.0, np.pi)
    params[nu:2 * nu] = (phases + rng.normal(0.0, 0.15, size=nu)) % (2.0 * np.pi)

    # Bias near zero
    BIAS_MAX = np.pi / 6.0
    params[2 * nu:3 * nu] = rng.uniform(-0.1 * BIAS_MAX, 0.1 * BIAS_MAX, size=nu)

    # freq, ramp near mid
    params[-2] = rng.uniform(0.6, 1.0)  # freq Hz
    params[-1] = rng.uniform(0.2, 0.5)  # ramp s^-1

    # clip into bounds
    np.clip(params, lo, hi, out=params)
    return params


# Globals set per episode
CURRENT_CPG_PARAMS: Optional[np.ndarray] = None
PREV_CTRL: Optional[np.ndarray] = None   # for smoothing


def cpg_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    """
    Controller callback that reads CURRENT_CPG_PARAMS and generates actions.
    - Initial quiet window (no actuation) to let things settle.
    - Exponential smoothing on control to avoid spikes.
    - Amplitude ramp and clipping to [-pi/2, pi/2].
    """
    nu = model.nu
    if nu == 0:
        return np.zeros(0, dtype=np.float64)

    assert CURRENT_CPG_PARAMS is not None, "CPG params not set!"
    params = CURRENT_CPG_PARAMS

    A = params[:nu]
    PHI = params[nu:2 * nu]
    B = params[2 * nu:3 * nu]
    freq = params[-2]
    ramp = params[-1]

    t = float(data.time)
    CONTROL_BOUND = np.pi / 2.0

    # Quiet start: no actuation for the first QUIET_TIME seconds
    if t < QUIET_TIME:
        u = np.zeros(nu, dtype=np.float64)
    else:
        # Smooth amplitude ramp to avoid early instability
        A_t = A * (1.0 - np.exp(-ramp * (t - QUIET_TIME)))
        raw = A_t * np.sin(2.0 * np.pi * freq * (t - QUIET_TIME) + PHI) + B

        # Exponential smoothing
        global PREV_CTRL
        if PREV_CTRL is None or PREV_CTRL.shape[0] != nu:
            PREV_CTRL = np.zeros(nu, dtype=np.float64)
        u = SMOOTH_ALPHA * PREV_CTRL + (1.0 - SMOOTH_ALPHA) * raw
        PREV_CTRL = u

    np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
    return u


# ----------------------------------------------------
# Genotype & decode
# ----------------------------------------------------
def random_genotype() -> List[np.ndarray]:
    """Three NDE input vectors (type, connection, rotation)."""
    return [
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
    ]


def decode_and_build(genotype: List[np.ndarray]):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core


# ----------------------------------------------------
# Fitness (UNCHANGED)
# ----------------------------------------------------
def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2
    )
    return -cartesian_distance


# ----------------------------------------------------
# Utilities: progress, duration schedule, probe/viability
# ----------------------------------------------------
def max_x_progress(history: list) -> float:
    if not history:
        return 0.0
    x0 = history[0][0]
    return max(p[0] - x0 for p in history)


def sim_duration_for_progress(best_prog: float) -> float:
    if best_prog >= CP2:
        return DUR_LONG
    if best_prog >= CP1:
        return DUR_MED
    return DUR_SHORT


def run_episode_with_params(
    core,
    params: np.ndarray,
    duration: float,
    steps_per_loop: int = 40,
    spawn_offset_z: float = 0.08,
) -> Tuple[float, list]:
    """
    Runs a MuJoCo episode with given CPG params; returns (fitness, xpos_history).
    """
    global CURRENT_CPG_PARAMS, PREV_CTRL
    CURRENT_CPG_PARAMS = params
    PREV_CTRL = None  # reset smoothing state for a fresh episode

    mj.set_mjcb_control(None)
    world = OlympicArena()
    spawn = [SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + spawn_offset_z]
    world.spawn(core.spec, spawn_position=spawn)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Quick actuator sanity: if no actuators, bail out
    if model.nu == 0:
        return -1e6, []

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=cpg_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    try:
        simple_runner(model, data, duration=duration, steps_per_loop=steps_per_loop)
    except Exception:
        return -1e6, []

    xpos_history = tracker.history.get("xpos", {})
    if len(xpos_history) == 0 or 0 not in xpos_history:
        return -1e6, []
    hist = xpos_history[0]
    if not np.isfinite(np.asarray(hist)).all():
        return -1e6, []
    return fitness_function(hist), hist


def is_viable_body(core) -> bool:
    """
    Actuator and probe-based viability filter to kill statues quickly.
    Gentler probe to reduce early instabilities.
    """
    # Check actuators exist
    tmp_world = OlympicArena()
    tmp_world.spawn(core.spec, spawn_position=[SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.08])
    tmp_model = tmp_world.spec.compile()
    if tmp_model.nu < 1:                           # loosened: allow any actuated body
        return False

    # Tiny-amp, very gentle probe
    nu = tmp_model.nu
    lo, hi = cpg_bounds(nu)
    probe = np.zeros_like(lo)

    # very small amps (down from 0.07*(pi/2) to 0.03*(pi/2))
    probe[:nu] = 0.03 * (np.pi / 2.0)
    # phases: alternating 0/pi
    phases = np.where((np.arange(nu) % 2) == 0, 0.0, np.pi)
    probe[nu:2 * nu] = phases
    # bias near zero
    probe[2 * nu:3 * nu] = 0.0
    # gentler freq & ramp
    probe[-2] = 0.5     # freq Hz (down from 0.8)
    probe[-1] = 0.15    # ramp s^-1 (down from 0.3)

    fit, hist = run_episode_with_params(core, probe, duration=PROBE_DURATION, steps_per_loop=40, spawn_offset_z=0.08)
    if hist == []:
        return False

    prog = max_x_progress(hist)
    return prog >= NONLEARNER_EPS


# ----------------------------------------------------
# Inner-loop optimizer: DE (RevDE-style parameters)
# ----------------------------------------------------
def de_optimize_cpg(
    core,
    budget_evals: int = 240,
    pop_size: int = 24,
    F: float = 0.5,
    CR: float = 0.9,
) -> Tuple[np.ndarray, float, list]:
    """
    Simple DE/rand/1/bin with RevDE-style defaults.
    Returns (best_params, best_fitness, best_hist).
    """
    # prepare bounds and init pop
    tmp_world = OlympicArena()
    tmp_world.spawn(core.spec, spawn_position=[SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.08])
    tmp_model = tmp_world.spec.compile()
    nu = tmp_model.nu
    lo, hi = cpg_bounds(nu)
    dim = lo.size

    # init population
    pop = np.vstack([cpg_init(nu, RNG) for _ in range(pop_size)])
    pop = np.clip(pop, lo, hi)

    # evaluate initial pop (short duration)
    best_prog = 0.0
    fits = []
    hists = []
    for i in range(pop_size):
        dur = sim_duration_for_progress(best_prog)
        f, hist = run_episode_with_params(core, pop[i], duration=dur, steps_per_loop=40, spawn_offset_z=0.08)
        fits.append(f)
        hists.append(hist)
        prog = max_x_progress(hist) if hist else 0.0
        best_prog = max(best_prog, prog)
    fits = np.asarray(fits, dtype=np.float64)

    evals_done = pop_size
    best_idx = int(np.argmax(fits))
    best_params = pop[best_idx].copy()
    best_fit = float(fits[best_idx])
    best_hist = hists[best_idx]

    # DE main loop
    while evals_done < budget_evals:
        dur = sim_duration_for_progress(best_prog)
        # one generation = pop_size trials
        for i in range(pop_size):
            if evals_done >= budget_evals:
                break

            r0, r1, r2 = RNG.choice(pop_size, size=3, replace=False)
            mutant = pop[r0] + F * (pop[r1] - pop[r2])

            cross_mask = RNG.random(dim) < CR
            if not np.any(cross_mask):
                cross_mask[RNG.integers(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])
            np.clip(trial, lo, hi, out=trial)

            f_trial, hist_trial = run_episode_with_params(core, trial, duration=dur, steps_per_loop=40, spawn_offset_z=0.08)
            evals_done += 1

            if f_trial >= fits[i]:
                pop[i] = trial
                fits[i] = f_trial
                hists[i] = hist_trial
                if f_trial >= best_fit:
                    best_fit = float(f_trial)
                    best_params = trial.copy()
                    best_hist = hist_trial
                prog = max_x_progress(hist_trial) if hist_trial else 0.0
                best_prog = max(best_prog, prog)

    return best_params, best_fit, best_hist


# ----------------------------------------------------
# Body suite construction
# ----------------------------------------------------
def sample_viable_bodies(n_bodies: int) -> List[Tuple[List[np.ndarray], Any]]:
    """
    Returns a list of tuples: (genotype, core) for viable bodies.
    """
    bodies = []
    attempts = 0
    while len(bodies) < n_bodies and attempts < MAX_ATTEMPTS:
        attempts += 1
        genotype = [
            RNG.random(GENE_LEN).astype(np.float32),
            RNG.random(GENE_LEN).astype(np.float32),
            RNG.random(GENE_LEN).astype(np.float32),
        ]
        try:
            robot_graph, core = decode_and_build(genotype)
            if is_viable_body(core):
                bodies.append((genotype, core))
                # save graph snapshot
                save_graph_as_json(robot_graph, OUTPUT / f"body_{len(bodies)}.json")

                # log nu safely (compile world, not core.spec directly)
                tmp_world = OlympicArena()
                tmp_world.spawn(core.spec, spawn_position=[SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.08])
                tmp_model = tmp_world.spec.compile()
                print(f"[suite] accepted body {len(bodies)} (nu={tmp_model.nu})")
        except Exception:
            continue
    return bodies


# ----------------------------------------------------
# Main PHASE 1 experiment
# ----------------------------------------------------
def main_phase1():
    print("=== PHASE 1: Inner-loop CPG + DE on a viable body suite ===")

    # Build a small suite of viable bodies
    suite = sample_viable_bodies(N_BODIES)
    if not suite:
        print("No viable bodies found. Try loosening filters or increasing attempts.")
        return

    for idx, (genotype, core) in enumerate(suite, start=1):
        print(f"\n--- Body {idx}/{len(suite)}: optimizing CPG with DE ---")
        best_params, best_fit, best_hist = de_optimize_cpg(
            core,
            budget_evals=240,   # ~10 generations at pop 24
            pop_size=24,
            F=0.5,
            CR=0.9,
        )
        print(f"[body {idx}] best fitness: {best_fit:.4f}")

        # Save best video (duration based on achieved progress)
        prog = max_x_progress(best_hist) if best_hist else 0.0
        final_dur = sim_duration_for_progress(prog)

        global CURRENT_CPG_PARAMS, PREV_CTRL
        CURRENT_CPG_PARAMS = best_params
        PREV_CTRL = None

        mj.set_mjcb_control(None)
        world = OlympicArena()
        spawn = [SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.08]
        world.spawn(core.spec, spawn_position=spawn)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(world.spec, data)

        ctrl = Controller(controller_callback_function=cpg_controller, tracker=tracker)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        video_folder = OUTPUT / f"videos_body_{idx}"
        video_folder.mkdir(exist_ok=True)
        recorder = VideoRecorder(output_folder=str(video_folder))
        video_renderer(model, data, duration=final_dur, video_recorder=recorder)
        print(f"[body {idx}] saved video to {video_folder}")

    print("\n=== PHASE 1 complete. Inner loop verified across suite. ===")


if __name__ == "__main__":
    main_phase1()
