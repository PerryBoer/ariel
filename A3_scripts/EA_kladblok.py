"""
Outer GA + Inner RevDE (single-script, single fitness)
- Operates ONLY on the three NDE input vectors (type, connection, rotation).
- One objective only: negative distance to TARGET_POSITION after rollout.
- Inner loop: small-pop RevDE to tune a simple CPG (phases per actuator + global A, f).
- No random-move probes; only two cheap gates that do NOT change objective:
  (1) compile + actuator band check; (2) 0.2s zero-control stability sanity.
- Biases and symmetry-aware operators on the three genotype vectors to increase viability.
- Roulette-wheel parent selection, one-point crossover per vector, Gaussian mutation.
- Genotype-level caching: if a body reappears, reuse its best fitness and CPG params.
- Saves best body JSON and a short video rendered with its learned controller.

You can toggle/elaborate budgets at the CONFIG section below.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import numpy.typing as npt
import mujoco as mj

# Clear any stale MuJoCo control callback
try:
    mj.set_mjcb_control(None)
except Exception:
    pass

# ---------- ARIEL imports ----------
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer

# ==================== CONFIG ====================
SEED = 42
RNG = np.random.default_rng(SEED)

# EA (outer) params
POP_SIZE = 10
GENS = 5
GENE_LEN = 64
NUM_MODULES = 30

# Controller limits / target & sim
CONTROL_BOUND = np.pi / 2
SIM_DURATION = 10.0              # rollout duration for fitness
QUIET_TIME = 0.60                # start-up quiet phase to avoid initial shocks
SPAWN_POS = [-0.8, 0, 0.1]       # per user request (no extra z-offset)
TARGET_POSITION = np.array([5.0, 0.0, 0.5], dtype=np.float64)

# Actuator band (quick viability gate, no secondary scoring)
NU_MIN, NU_MAX = 4, 20

# EA operators
CROSSOVER_PROB = 0.80
MUTATION_PROB = 0.10
MUTATION_STD = 0.10

# Symmetry options
SYMMETRIZE_P = 0.10              # occasional pairwise symmetrization

# Inner RevDE budgets
REVDE_POP = 16
REVDE_GENS = 12
REVDE_F = 0.6                    # differential weight
REVDE_CR = 0.7                   # crossover rate
REVDE_SIM = 8.0                  # (seconds) inner-loop rollout (can be <= SIM_DURATION)

# Paths
CWD = Path.cwd()
OUTPUT = CWD / "__output__"
(OUTPUT / "videos").mkdir(parents=True, exist_ok=True)

# =============== Helpers / hashing ===============
def hash_genotype(gen: List[np.ndarray]) -> bytes:
    parts = [np.ascontiguousarray(v).astype(np.float32).tobytes() for v in gen]
    return b"|".join(parts)

# genotype -> cache: fitness + best controller params
_eval_cache: Dict[bytes, Dict[str, Any]] = {}

# =============== Init with legal biases ===============
def biased_random_genotype() -> List[np.ndarray]:
    """Biased priors to improve actuator/connectivity emergence (still evolutionary & legal)."""
    # type: bias upward (actuated modules more likely)
    type_vec = np.clip(0.80 + 0.20 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)
    # connection: front-load density (decay across indices) + noise
    base = np.power(0.85, np.arange(GENE_LEN, dtype=np.float32))
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    conn_vec = np.clip(base + 0.15 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)
    # rotation: bimodal {0,1} + noise for bilateral hints
    modes = RNG.integers(0, 2, size=GENE_LEN).astype(np.float32)
    rot_vec = np.clip(modes + 0.18 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)
    return [type_vec, conn_vec, rot_vec]

# =============== GA operators ===============
def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = a.size
    point = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:point], b[point:]]).astype(np.float32)
    c2 = np.concatenate([b[:point], a[point:]]).astype(np.float32)
    return c1, c2


def crossover_per_chromosome(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    child1, child2 = [], []
    for idx in range(3):
        if RNG.random() < CROSSOVER_PROB:
            ca, cb = one_point_crossover(pa[idx], pb[idx])
        else:
            ca, cb = pa[idx].copy(), pb[idx].copy()
        child1.append(ca); child2.append(cb)
    return child1, child2


def symmetrize_pairs(vec: np.ndarray, p: float = SYMMETRIZE_P) -> np.ndarray:
    L = vec.size
    out = vec.copy()
    if RNG.random() < p:
        for i in range(L // 2):
            j = L - 1 - i
            m = 0.5 * (out[i] + out[j])
            out[i] = m; out[j] = m
    return out


def symmetry_aware_mutation(vec: np.ndarray, mut_p: float, sigma: float) -> np.ndarray:
    L = vec.size
    out = vec.copy()
    for i in range(L // 2):
        if RNG.random() < mut_p:
            j = L - 1 - i
            eps = RNG.normal(0.0, sigma)
            out[i] = np.clip(out[i] + eps, 0.0, 1.0)
            out[j] = np.clip(out[j] + eps, 0.0, 1.0)
    if L % 2 == 1 and RNG.random() < mut_p:
        mid = L // 2
        out[mid] = np.clip(out[mid] + RNG.normal(0.0, sigma), 0.0, 1.0)
    return out.astype(np.float32)


def gaussian_mutation(gen: List[np.ndarray]) -> List[np.ndarray]:
    mutated = []
    for chrom in gen:
        # standard per-gene mutate mask, then enforce symmetry-aware coupling
        to_mut = RNG.random(chrom.shape) < MUTATION_PROB
        noise = RNG.normal(loc=0.0, scale=MUTATION_STD, size=chrom.shape).astype(np.float32)
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0)
        new_chrom = symmetry_aware_mutation(new_chrom, mut_p=0.05, sigma=MUTATION_STD)
        new_chrom = symmetrize_pairs(new_chrom)
        mutated.append(new_chrom.astype(np.float32))
    return mutated

# =============== Decode & build ===============
def decode_and_build(genotype: List[np.ndarray]):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core

# =============== Fitness ===============
def fitness_function(history: list[Tuple[float, float, float]]) -> float:
    """Negative Euclidean distance to TARGET_POSITION (maximization)."""
    if not history:
        return -1e6
    xc, yc, zc = history[-1]
    diff = TARGET_POSITION - np.array([xc, yc, zc], dtype=np.float64)
    return -float(np.linalg.norm(diff))

# =============== Zero-control stability sanity ===============
def zero_control_stability(model: mj.MjModel, data: mj.MjData, duration: float = 0.2) -> bool:
    """Run a very short sim with zero control; return False if it NaNs/explodes."""
    mj.set_mjcb_control(None)
    def cb(m: mj.MjModel, d: mj.MjData):
        if m.nu:
            d.ctrl[:] = 0.0
    mj.set_mjcb_control(cb)
    try:
        simple_runner(model, data, duration=duration, steps_per_loop=40)
    except Exception:
        return False
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return False
    return True

# =============== Inner RevDE (CPG) ===============

def make_controller_callback(model: mj.MjModel, phases: np.ndarray, A: float, freq_hz: float):
    """Return a MuJoCo control callback implementing the CPG with given params."""
    nu = int(model.nu)
    phases = phases.astype(np.float64)
    A = float(np.clip(A, 0.0, CONTROL_BOUND))
    FREQ_HZ = float(np.clip(freq_hz, 0.3, 1.0))
    OMEGA = 2.0 * np.pi * FREQ_HZ

    def cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME:
            if m.nu:
                d.ctrl[:] = 0.0
            return
        t = d.time - QUIET_TIME
        # per-actuator sine with phase offsets
        u = A * np.sin(OMEGA * t + phases)
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    return cb


def clamp_params(phases: np.ndarray, A: float, f: float) -> Tuple[np.ndarray, float, float]:
    # wrap phases to [0, 2π)
    phases = np.mod(phases, 2.0 * np.pi)
    A = float(np.clip(A, 0.0, CONTROL_BOUND))
    f = float(np.clip(f, 0.3, 1.0))
    return phases, A, f


def revde_optimize_cpg(spec, model: mj.MjModel,
                       sim_seconds: float = REVDE_SIM) -> Tuple[float, Dict[str, Any]]:
    """Small-pop RevDE optimizing phases (per-actuator) + global amplitude + frequency.
    Returns (best_fitness, {phases, A, f}).
    Note: Creates a fresh Tracker and MjData per candidate to avoid binding issues.
    """
    nu = int(model.nu)
    if nu == 0:
        return -1e6, {"phases": np.zeros(0), "A": 0.0, "f": 0.7}

    D = nu + 2  # phases per actuator + (A, f)

    # Bounds / initialization
    def init_individual():
        phases0 = RNG.uniform(0.0, 2.0*np.pi, size=nu).astype(np.float64)
        A0 = float(RNG.uniform(0.2, 0.8) * CONTROL_BOUND)
        f0 = float(RNG.uniform(0.4, 0.9))
        return np.concatenate([phases0, np.array([A0, f0])]).astype(np.float64)

    pop = np.stack([init_individual() for _ in range(REVDE_POP)], axis=0)

    def evaluate(vec: np.ndarray) -> float:
        phases = vec[:nu].copy()
        A, f = float(vec[-2]), float(vec[-1])
        phases, A, f = clamp_params(phases, A, f)

        # fresh data + tracker per evaluation
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        local_tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        local_tracker.setup(spec, data)

        mj.set_mjcb_control(None)
        cb = make_controller_callback(model, phases, A, f)
        mj.set_mjcb_control(cb)

        try:
            simple_runner(model, data, duration=sim_seconds, steps_per_loop=80)
        except Exception:
            return -1e6

        xpos = local_tracker.history.get("xpos", {}).get(0, [])
        return fitness_function(xpos)

    fitness = np.array([evaluate(ind) for ind in pop], dtype=np.float64)

    for _ in range(REVDE_GENS):
        best_idx = int(np.argmax(fitness))
        best = pop[best_idx]
        new_pop = pop.copy()
        new_fit = fitness.copy()
        for i in range(REVDE_POP):
            # choose r1, r2 distinct from i
            idxs = [j for j in range(REVDE_POP) if j != i]
            r1, r2 = RNG.choice(idxs, size=2, replace=False)
            vi = pop[i] + REVDE_F * (best - pop[i]) + REVDE_F * (pop[r1] - pop[r2])
            # binomial crossover
            cross_mask = RNG.random(D) < REVDE_CR
            if not cross_mask.any():
                cross_mask[RNG.integers(0, D)] = True
            trial = np.where(cross_mask, vi, pop[i])
            fit_trial = evaluate(trial)
            if fit_trial > fitness[i]:
                new_pop[i] = trial
                new_fit[i] = fit_trial
        pop, fitness = new_pop, new_fit

    best_idx = int(np.argmax(fitness))
    best_vec = pop[best_idx]
    phases = best_vec[:nu].copy()
    A, f = float(best_vec[-2]), float(best_vec[-1])
    phases, A, f = clamp_params(phases, A, f)
    return float(fitness[best_idx]), {"phases": phases, "A": A, "f": f}

# =============== Evaluate a genotype (single objective) ===============
 #(single objective) ===============
def evaluate_genotype(genotype: List[np.ndarray]) -> Tuple[float, list, Dict[str, Any]]:
    key = hash_genotype(genotype)
    if key in _eval_cache:
        c = _eval_cache[key]
        return c["fitness"], c.get("history", []), c.get("ctrl", {})

    # Decode & compile
    try:
        robot_graph, core = decode_and_build(genotype)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
    except Exception:
        _eval_cache[key] = {"fitness": -1e6}
        return -1e6, [], {}

    # Actuator band gate
    nu = int(model.nu)
    if nu < NU_MIN or nu > NU_MAX:
        _eval_cache[key] = {"fitness": -1e6}
        return -1e6, [], {}

    # Zero-control stability sanity (very cheap)
    if not zero_control_stability(model, data, duration=0.2):
        _eval_cache[key] = {"fitness": -1e6}
        return -1e6, [], {}

    # Inner RevDE to get controller params and best fitness (single score)
    best_fit, ctrl_params = revde_optimize_cpg(world.spec, model, sim_seconds=REVDE_SIM)

    # Optional: final short full-duration confirmation rollout for history (uses learned params)
    phases = ctrl_params.get("phases", np.zeros(nu))
    A = ctrl_params.get("A", 0.4 * CONTROL_BOUND)
    f = ctrl_params.get("f", 0.7)
    phases, A, f = clamp_params(phases, A, f)

    mj.set_mjcb_control(None)
    data2 = mj.MjData(model)
    mj.mj_resetData(model, data2)
    tracker2 = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker2.setup(world.spec, data2)
    cb = make_controller_callback(model, phases, A, f)
    mj.set_mjcb_control(cb)
    try:
        simple_runner(model, data2, duration=SIM_DURATION, steps_per_loop=80)
    except Exception:
        pass

    hist = tracker2.history.get("xpos", {}).get(0, [])
    fit_final = fitness_function(hist) if hist else best_fit

    rec = {
        "fitness": fit_final,
        "history": hist,
        "ctrl": ctrl_params,
        "robot_graph": robot_graph,
        "core": core,
    }
    _eval_cache[key] = rec
    return fit_final, hist, ctrl_params

# =============== EA main loop ===============
def run_ea():
    mj.set_mjcb_control(None)
    population = [biased_random_genotype() for _ in range(POP_SIZE)]

    fitnesses: List[float] = []
    histories: List[list] = []
    ctrls: List[Dict[str, Any]] = []

    for i, gen in enumerate(population):
        fit, his, ctrl = evaluate_genotype(gen)
        fitnesses.append(fit); histories.append(his); ctrls.append(ctrl)

    for gen_idx in range(GENS):
        print(f"=== Generation {gen_idx+1} ===")

        # roulette-wheel selection probabilities
        f_arr = np.array(fitnesses, dtype=np.float64)
        weights = f_arr - f_arr.min() + 1e-9
        probs = weights / max(weights.sum(), 1e-9)

        # produce children
        children: List[List[np.ndarray]] = []
        while len(children) < POP_SIZE:
            pa, pb = RNG.choice(len(population), size=2, replace=False, p=probs)
            c1, c2 = crossover_per_chromosome(population[pa], population[pb])
            children.append(gaussian_mutation(c1))
            if len(children) < POP_SIZE:
                children.append(gaussian_mutation(c2))

        # evaluate children
        child_fits: List[float] = []
        child_hists: List[list] = []
        child_ctrls: List[Dict[str, Any]] = []
        for chi in children:
            fit, his, ctrl = evaluate_genotype(chi)
            child_fits.append(fit); child_hists.append(his); child_ctrls.append(ctrl)

        # elitist survivor selection (mu+lambda) → keep best POP_SIZE
        combined = population + children
        combined_fit = fitnesses + child_fits
        combined_hist = histories + child_hists
        combined_ctrl = ctrls + child_ctrls
        order = np.argsort(combined_fit)[::-1][:POP_SIZE]
        population = [combined[i] for i in order]
        fitnesses = [combined_fit[i] for i in order]
        histories = [combined_hist[i] for i in order]
        ctrls = [combined_ctrl[i] for i in order]

        print("Survivors (top -> bottom):")
        for r, f in enumerate(fitnesses):
            print(f"{r+1}: {f:.4f}")

    # Best individual
    best_idx = int(np.argmax(fitnesses))
    best_gen, best_hist, best_fit, best_ctrl = (
        population[best_idx], histories[best_idx], fitnesses[best_idx], ctrls[best_idx]
    )
    print("=== EA finished ===")
    print(f"Best fitness: {best_fit:.4f}")

    # Save best robot JSON
    robot_graph, core = decode_and_build(best_gen)
    out_json = OUTPUT / "best_robot.json"
    save_graph_as_json(robot_graph, out_json)
    print(f"Saved best robot graph to {out_json}")

    # Save best video with learned controller
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    nu = int(model.nu)
    phases = np.zeros(nu) if not best_ctrl else best_ctrl.get("phases", np.zeros(nu))
    A = CONTROL_BOUND * 0.4 if not best_ctrl else best_ctrl.get("A", CONTROL_BOUND*0.4)
    f = 0.7 if not best_ctrl else best_ctrl.get("f", 0.7)
    phases, A, f = clamp_params(phases, A, f)

    cb = make_controller_callback(model, phases, A, f)
    mj.set_mjcb_control(cb)

    video_folder = OUTPUT / "videos"
    recorder = VideoRecorder(output_folder=str(video_folder))
    video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    print(f"Saved video of best robot to {video_folder}")

    return best_gen, best_hist, best_fit


if __name__ == "__main__":
    run_ea()
