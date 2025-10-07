"""
Evolutionary algorithm to evolve the robot body. Performs:
- roulette wheel parent selection
- one-point crossover per genotype vector
- gaussian mutation per genotype vector
- non-learner filter (short probe to discard statues)
- dynamic simulation duration per generation
Saves best robot json and video after final generation.
"""

from pathlib import Path
from typing import Any, List, Tuple

import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# ---- RNG ----
SEED = 42
RNG = np.random.default_rng(SEED)

# ---- EA params ----
POP_SIZE = 10
GENS = 30
GENE_LEN = 64
MUTATION_STD = 0.1
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.8

# ---- World / robot params ----
SPAWN_POS = [-0.8, 0.0, 0.10]  # base; we add +0.05 z in spawns to avoid clipping
NUM_MODULES = 30

# ---- Evaluation / plotting ----
# Target is used by *your* fitness; unchanged:
TARGET_POSITION = [5.0, 0.0, 0.5]

# Base/default duration (used for final video and as a fallback)
SIM_DURATION = 10.0

# Dynamic-duration checkpoints (forward progress from spawn, along x)
CP1 = 1.5     # approx end of early smooth part
CP2 = 4.5     # through rugged part
DUR_SHORT = 12.0
DUR_MED   = 35.0
DUR_LONG  = 80.0

# Non-learner probe (discard statues fast)
PROBE_DURATION = 2.5
NONLEARNER_EPS = 0.10  # meters forward progress required in probe

# ---- I/O ----
CWD = Path.cwd()
OUTPUT = CWD / "__output__"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Controller: Fixed CPG (global sine) with alternation + amplitude ramp
# ======================================================================
def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    """
    Fixed-parameter CPG:
    - Phase alternation across actuators (even=0, odd=pi)
    - Amplitude ramp A(t) to avoid early instability
    - Hard clipping to [-pi/2, pi/2]
    """
    nu: int = model.nu
    if nu == 0:
        return np.zeros(0, dtype=np.float64)

    t: float = float(data.time)

    CONTROL_BOUND = np.pi / 2
    A_MAX = 0.45 * CONTROL_BOUND     # steady-state amplitude (conservative to reduce QACC explosions)
    A0 = 0.12 * CONTROL_BOUND        # initial small amplitude
    RAMP_RATE = 0.30                 # seconds^-1; smooth ramp
    FREQ_HZ = 0.8                    # gait frequency in Hz (tame to reduce instability)
    OMEGA = 2.0 * np.pi * FREQ_HZ
    BIAS = 0.0

    # Smooth amplitude ramp
    A_t = A0 + (A_MAX - A0) * (1.0 - np.exp(-RAMP_RATE * t))

    # Alternating phases: even indices 0.0, odd indices pi
    idx = np.arange(nu, dtype=np.float64)
    phases = np.where((idx % 2) == 0, 0.0, np.pi)

    # CPG signal
    u = A_t * np.sin(OMEGA * t + phases) + BIAS

    # Hard bounds
    np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
    return u


# ======================================================================
# Genotype helpers (3 NDE input vectors only)
# ======================================================================
def random_genotype() -> List[np.ndarray]:
    """Initialize random genotype vectors (type, connection, rotation)."""
    return [
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
    ]


def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = a.size
    point = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1.astype(np.float32), c2.astype(np.float32)


def crossover_per_chromosome(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    child1, child2 = [], []
    for idx in range(3):
        if RNG.random() < CROSSOVER_PROB:
            ca, cb = one_point_crossover(pa[idx], pb[idx])
            child1.append(ca)
            child2.append(cb)
        else:
            child1.append(pa[idx].copy())
            child2.append(pb[idx].copy())
    return child1, child2


def gaussian_mutation(gen: List[np.ndarray]) -> List[np.ndarray]:
    """Gaussian mutation gene-wise with clipping to [0,1]."""
    mutated = []
    for chrom in gen:
        to_mut = RNG.random(chrom.shape) < MUTATION_PROB
        noise = RNG.normal(loc=0.0, scale=MUTATION_STD, size=chrom.shape).astype(np.float32)
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0)
        mutated.append(new_chrom.astype(np.float32))
    return mutated


# ======================================================================
# Decode genotype to body
# ======================================================================
def decode_and_build(genotype: List[np.ndarray]):
    """Decode NDE from genotype and build robot."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core


# ======================================================================
# Fitness (UNCHANGED as requested)
# ======================================================================
def fitness_function(history: list[float]) -> float:
    """From template: negative cartesian distance to TARGET_POSITION (maximization)."""
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance


# ======================================================================
# Utilities for dynamic duration and viability
# ======================================================================
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


def is_viable_learner(genotype: List[np.ndarray]) -> bool:
    """
    Short probe run to discard statues (non-learners).
    Uses the same fixed CPG. Slightly higher spawn z to avoid initial clipping.
    """
    try:
        robot_graph, core = decode_and_build(genotype)

        mj.set_mjcb_control(None)
        world = OlympicArena()
        spawn = [SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.05]
        world.spawn(core.spec, spawn_position=spawn)

        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(world.spec, data)

        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # Smaller steps_per_loop can help with early instability
        simple_runner(model, data, duration=PROBE_DURATION, steps_per_loop=50)

        hist = tracker.history.get("xpos", {}).get(0, None)
        if hist is None or len(hist) < 2:
            return False

        x0 = hist[0][0]
        x1 = hist[-1][0]
        return (x1 - x0) >= NONLEARNER_EPS

    except Exception as e:
        # Any failure counts as non-viable
        return False


def make_viable_random(max_attempts: int = 12) -> List[np.ndarray]:
    attempts = 0
    g = random_genotype()
    while attempts < max_attempts and not is_viable_learner(g):
        g = random_genotype()
        attempts += 1
    return g


# ======================================================================
# Evaluation
# ======================================================================
def evaluate_genotype(genotype: List[np.ndarray], duration: float) -> Tuple[float, list]:
    """Run headless simulation with CPG and calculate fitness."""
    try:
        robot_graph, core = decode_and_build(genotype)

        mj.set_mjcb_control(None)
        world = OlympicArena()
        spawn = [SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.05]
        world.spawn(core.spec, spawn_position=spawn)

        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(world.spec, data)

        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        simple_runner(model, data, duration=duration, steps_per_loop=50)

        xpos_history = tracker.history.get("xpos", {})
        if len(xpos_history) == 0 or 0 not in xpos_history:
            return -1e6, []

        hist = xpos_history[0]
        # Guard against NaNs in positions
        if not np.isfinite(np.asarray(hist)).all():
            return -1e6, []

        return fitness_function(hist), hist

    except Exception as e:
        # Any sim failure → very low fitness
        return -1e6, []


# ======================================================================
# EA loop
# ======================================================================
def run_ea():
    mj.set_mjcb_control(None)

    # Initial population with viability filter
    population = [make_viable_random() for _ in range(POP_SIZE)]
    fitnesses, histories = [], []

    # Initial evaluation with short duration
    duration_this_gen = DUR_SHORT
    for gen in population:
        fit, his = evaluate_genotype(gen, duration=duration_this_gen)
        fitnesses.append(fit)
        histories.append(his)

    for gen_idx in range(GENS):
        print(f"\n=== Generation {gen_idx+1} ===")

        # Update duration based on best progress so far
        try:
            best_prog = max(max_x_progress(h) for h in histories if h)
        except ValueError:
            best_prog = 0.0
        duration_this_gen = sim_duration_for_progress(best_prog)

        # Roulette selection probabilities
        f_arr = np.array(fitnesses, dtype=np.float64)
        # Shift to positive; if all equal, uniform
        w = f_arr - f_arr.min() + 1e-6
        if not np.isfinite(w).all() or w.sum() <= 0:
            probs = np.ones_like(w) / len(w)
        else:
            probs = w / w.sum()

        # Create children
        children = []
        while len(children) < POP_SIZE:
            pa, pb = RNG.choice(len(population), size=2, replace=False, p=probs)
            c1, c2 = crossover_per_chromosome(population[pa], population[pb])
            c1 = gaussian_mutation(c1)
            c2 = gaussian_mutation(c2)

            # Viability check with small retry loop
            attempts = 0
            while not is_viable_learner(c1) and attempts < 8:
                c1 = gaussian_mutation(c1)
                attempts += 1
            attempts = 0
            while not is_viable_learner(c2) and attempts < 8:
                c2 = gaussian_mutation(c2)
                attempts += 1

            children.append(c1)
            if len(children) < POP_SIZE:
                children.append(c2)

        # Evaluate children with current duration
        child_fitnesses, child_histories = [], []
        for chi in children:
            fit, his = evaluate_genotype(chi, duration=duration_this_gen)
            child_fitnesses.append(fit)
            child_histories.append(his)

        # Elitist survivor selection
        combined = population + children
        combined_fit = fitnesses + child_fitnesses
        combined_hist = histories + child_histories

        order = np.argsort(combined_fit)[::-1][:POP_SIZE]
        population = [combined[i] for i in order]
        fitnesses = [combined_fit[i] for i in order]
        histories = [combined_hist[i] for i in order]

        print("Survivors:")
        for r, f in enumerate(fitnesses):
            print(f"{r+1}: {f:.4f}")

    # Final artifacts
    best_idx = int(np.argmax(fitnesses))
    best_gen, best_hist, best_fit = population[best_idx], histories[best_idx], fitnesses[best_idx]
    print("\n=== EA finished ===")
    print(f"Best fitness: {best_fit:.4f}")

    # Save best robot graph
    robot_graph, core = decode_and_build(best_gen)
    save_graph_as_json(robot_graph, OUTPUT / "best_robot.json")
    print(f"Saved best robot graph to {OUTPUT/'best_robot.json'}")

    # Save video of best robot (use longer duration for clarity)
    mj.set_mjcb_control(None)
    world = OlympicArena()
    spawn = [SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2] + 0.05]
    world.spawn(core.spec, spawn_position=spawn)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    video_folder = OUTPUT / "videos"
    video_folder.mkdir(exist_ok=True)
    recorder = VideoRecorder(output_folder=str(video_folder))
    # Use the same duration policy for final render, but at least MED
    final_prog = max_x_progress(best_hist) if best_hist else 0.0
    final_duration = max(DUR_MED, sim_duration_for_progress(final_prog))
    video_renderer(model, data, duration=final_duration, video_recorder=recorder)
    print(f"Saved video of best robot to {video_folder}")

    return best_gen, best_hist, best_fit


if __name__ == "__main__":
    run_ea()
