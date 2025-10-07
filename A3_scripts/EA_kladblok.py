"""
Outer-loop GA for evolving viable walking bodies (legal priors + filters).
- Operates ONLY on the three NDE input vectors (type, connection, rotation).
- Staged evaluation: static checks -> random-moves probe -> gentle CPG probe.
- Uses a surrogate viability score for selection in early gens, then switches to true fitness.
- Tournament selection, symmetry-aware mutation, elitism, caching.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import numpy.typing as npt
import mujoco as mj

# IMPORTANT: clear MuJoCo control callback immediately (prevents engine "Python exception"/segfault)
try:
    mj.set_mjcb_control(None)  # DO NOT REMOVE
except Exception:
    pass

# ---------- ARIEL imports ----------
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

# ========= Globals / config =========
SEED = 42
RNG = np.random.default_rng(SEED)

# Paths
CWD = Path.cwd()
OUT = CWD / "__outer_ga__"
(OUT / "videos").mkdir(parents=True, exist_ok=True)

# Arena / task
TARGET_POS = np.array([5.0, 0.0, 0.5])   # forward is +X in OlympicArena
SPAWN_BASE = np.array([-0.8, 0.0, 0.10])
SPAWN_Z_OFFSET = 0.20                    # lifted spawn to avoid interpenetration

# NDE + genotype
NUM_MODULES = 30
GENE_LEN = 64     # per vector
CHROMOSOMES = 3   # [type, connection, rotation]

# GA
POP = 120
GENS = 40
ELITES = 2
TOURNAMENT_K = 4
MUT_P = 0.12
MUT_SIGMA = 0.12
SIGMA_DRIFT = 0.10      # self-adaptive log-normal drift
CROSSOVER_P = 0.80
BLEND_XOVER_P = 0.20    # sometimes use arithmetic/blend xover (rot/type)
CURRICULUM_GENS = 12    # use viability score for selection for first N gens

# Viability filters / probes
NU_MIN, NU_MAX = 4, 20
RAND_PROBE_DUR = 1.0
RAND_PROBE_STD = 0.02
PROBE_DUR = 2.5
VIAB_EPS1 = 0.01    # random-moves forward progress
VIAB_EPS2 = 0.05    # gentle CPG forward progress

# Final video/eval duration
FINAL_DUR = 12.0

# Caching
_eval_cache: Dict[bytes, Dict[str, Any]] = {}

# Controller hygiene
QUIET_TIME = 0.60
CONTROL_BOUND = np.pi / 2

# ========= Helpers =========
def hash_genotype(gen: List[np.ndarray]) -> bytes:
    # Stable hashable key for caching
    return b"|".join([np.ascontiguousarray(v).tobytes() for v in gen])

def init_biased_population(n: int) -> List[List[np.ndarray]]:
    """Legal priors entirely in the three input vectors."""
    pop: List[List[np.ndarray]] = []
    for _ in range(n):
        # type: bias toward hinge-able mix -> mean ~0.65 with noise
        type_vec = np.clip(0.65 + 0.18 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)

        # connection: front-loaded (shorter connections early indices) + noise
        base = np.power(0.90, np.arange(GENE_LEN, dtype=np.float32))  # decays with index
        base = (base - base.min()) / (base.max() - base.min() + 1e-9)
        conn_vec = np.clip(base + 0.12 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)

        # rotation: bimodal around {0, 1} with noise -> encourages mirrored/symmetric
        modes = RNG.integers(0, 2, size=GENE_LEN).astype(np.float32)
        rot_vec = np.clip(modes + 0.18 * RNG.standard_normal(GENE_LEN), 0.0, 1.0).astype(np.float32)

        pop.append([type_vec, conn_vec, rot_vec])
    return pop

def symmetrize_pairs(vec: np.ndarray, p: float = 0.10) -> np.ndarray:
    """Occasionally average index i with its 'mirror' (midpoint pairing)."""
    L = vec.size
    out = vec.copy()
    if RNG.random() < p:
        for i in range(L // 2):
            j = L - 1 - i
            m = 0.5 * (out[i] + out[j])
            out[i] = m; out[j] = m
    return out

def symmetry_aware_mutation(vec: np.ndarray, mut_p: float, sigma: float) -> np.ndarray:
    """Same noise to mirrored indices to preserve bilateral features."""
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

def crossover_one_point(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = a.size
    point = int(RNG.integers(1, L))
    return (np.concatenate([a[:point], b[point:]]).astype(np.float32),
            np.concatenate([b[:point], a[point:]]).astype(np.float32))

def crossover_blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    lam = RNG.uniform(0.3, 0.7) if alpha == 0.5 else alpha
    c1 = np.clip(lam * a + (1 - lam) * b, 0.0, 1.0).astype(np.float32)
    c2 = np.clip(lam * b + (1 - lam) * a, 0.0, 1.0).astype(np.float32)
    return c1, c2

def build_graph(gen: List[np.ndarray]):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_type, p_conn, p_rot = nde.forward(gen)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    return hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)

def _spawn_world(robot_graph, z_off: float):
    core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    spawn = SPAWN_BASE + np.array([0.0, 0.0, z_off])
    world.spawn(core.spec, spawn_position=spawn.tolist())
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    return world, model, data

# ========= Probes & fitness =========
def random_moves_probe(robot_graph) -> float:
    """Tiny random sinusoid per actuator, QUIET start."""
    mj.set_mjcb_control(None)
    world, model, data = _spawn_world(robot_graph, SPAWN_Z_OFFSET)
    if model.nu == 0: return 0.0

    phases = RNG.uniform(0, 2*np.pi, size=model.nu)
    def cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME: 
            if m.nu: d.ctrl[:] = 0.0
            return
        u = RAND_PROBE_STD * np.sin(2*np.pi*0.6*(d.time - QUIET_TIME) + phases)
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    mj.set_mjcb_control(cb)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    try:
        simple_runner(model, data, duration=RAND_PROBE_DUR, steps_per_loop=40)
    except Exception:
        return 0.0

    xpos = tracker.history.get("xpos", {}).get(0, [])
    if len(xpos) == 0: return 0.0
    x0 = xpos[0][0]
    return max(p[0]-x0 for p in xpos)

def gentle_cpg_probe(robot_graph) -> float:
    """Alternating phases, very small amplitude, QUIET + ramp."""
    mj.set_mjcb_control(None)
    world, model, data = _spawn_world(robot_graph, SPAWN_Z_OFFSET)
    if model.nu == 0: return 0.0
    nu = model.nu

    A = 0.01 * CONTROL_BOUND
    PHI = np.where((np.arange(nu) % 2) == 0, 0.0, np.pi)
    FREQ = 0.4
    RAMP = 0.10

    def cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME:
            if m.nu: d.ctrl[:] = 0.0
            return
        t = d.time - QUIET_TIME
        A_t = A * (1.0 - np.exp(-RAMP * t))
        u = A_t * np.sin(2*np.pi*FREQ*t + PHI)
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    mj.set_mjcb_control(cb)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    try:
        simple_runner(model, data, duration=PROBE_DUR, steps_per_loop=40)
    except Exception:
        return 0.0

    xpos = tracker.history.get("xpos", {}).get(0, [])
    if len(xpos) == 0: return 0.0
    x0 = xpos[0][0]
    return max(p[0]-x0 for p in xpos)

def true_fitness(robot_graph, duration: float = FINAL_DUR) -> float:
    """Assignment fitness (negative distance to TARGET_POS) using a slightly stronger CPG for evaluation only."""
    mj.set_mjcb_control(None)
    world, model, data = _spawn_world(robot_graph, SPAWN_Z_OFFSET)
    if model.nu == 0: return -1e6
    nu = model.nu

    # Safe global-CPG for evaluation (still gentle)
    A = 0.30 * CONTROL_BOUND
    FREQ = 0.7
    RAMP = 0.8
    PHASES = 2*np.pi * np.arange(nu)/max(1,nu)

    def cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME:
            if m.nu: d.ctrl[:] = 0.0; return
        t = d.time - QUIET_TIME
        A_t = A*(1 - np.exp(-RAMP*t))
        u = A_t * np.sin(2*np.pi*FREQ*t + PHASES)
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    mj.set_mjcb_control(cb)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    try:
        simple_runner(model, data, duration=duration, steps_per_loop=80)
    except Exception:
        return -1e6

    hist = tracker.history.get("xpos", {}).get(0, [])
    if len(hist) == 0: return -1e6
    end = np.array(hist[-1])
    return -float(np.linalg.norm(TARGET_POS - end))

# ========= Viability evaluation with caching =========
def evaluate_viability(gen: List[np.ndarray]) -> Dict[str, Any]:
    key = hash_genotype(gen)
    if key in _eval_cache:
        return _eval_cache[key]

    result = {
        "nu": 0,
        "static_ok": False,
        "rand_prog": 0.0,
        "probe_prog": 0.0,
        "viab_score": 0.0,
        "fitness": -1e6,
        "robot_graph": None,
    }

    # Decode -> static checks
    try:
        robot_graph = build_graph(gen)
        result["robot_graph"] = robot_graph
    except Exception:
        _eval_cache[key] = result
        return result

    # compile once to read nu
    try:
        world, model, _ = _spawn_world(robot_graph, SPAWN_Z_OFFSET)
        nu = int(model.nu)
    except Exception:
        _eval_cache[key] = result
        return result

    result["nu"] = nu
    if not (NU_MIN <= nu <= NU_MAX):
        _eval_cache[key] = result
        return result
    result["static_ok"] = True

    # Random-moves probe
    rp = random_moves_probe(robot_graph)
    result["rand_prog"] = rp
    if rp < VIAB_EPS1:
        _eval_cache[key] = result
        return result

    # Gentle CPG probe
    gp = gentle_cpg_probe(robot_graph)
    result["probe_prog"] = gp
    if gp < VIAB_EPS2:
        _eval_cache[key] = result
        return result

    # Viability score (simple blend; can be tuned)
    result["viab_score"] = 0.4 * rp + 0.6 * gp

    # True fitness (still cheap fixed CPG; later swap for inner CPG+DE best fit)
    result["fitness"] = true_fitness(robot_graph, duration=FINAL_DUR)

    _eval_cache[key] = result
    return result

# ========= GA operators =========
def parents_tournament(pop: List[List[np.ndarray]], scores: np.ndarray, k: int) -> Tuple[int, int]:
    cand = RNG.choice(len(pop), size=(2, k), replace=False)
    i = cand[0, np.argmax(scores[cand[0]])]
    j = cand[1, np.argmax(scores[cand[1]])]
    return int(i), int(j)

def make_children(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    child1, child2 = [], []
    for idx in range(CHROMOSOMES):
        a, b = pa[idx], pb[idx]
        if RNG.random() < CROSSOVER_P:
            if idx != 1 and RNG.random() < BLEND_XOVER_P:  # blend occasionally (type/rot)
                ca, cb = crossover_blend(a, b)
            else:
                ca, cb = crossover_one_point(a, b)
        else:
            ca, cb = a.copy(), b.copy()

        # symmetry-aware mutation with self-adaptive sigma
        # drift sigma per vector
        sigma = MUT_SIGMA * np.exp(SIGMA_DRIFT * RNG.standard_normal())
        ca = symmetry_aware_mutation(ca, MUT_P, sigma)
        cb = symmetry_aware_mutation(cb, MUT_P, sigma)

        # occasional symmetrize
        ca = symmetrize_pairs(ca, p=0.10)
        cb = symmetrize_pairs(cb, p=0.10)

        child1.append(ca)
        child2.append(cb)
    return child1, child2

# ========= Main loop =========
def run_outer_ga():
    print("=== Outer GA: viable body evolution ===")
    population = init_biased_population(POP)

    # Evaluate initial pop
    evals = [evaluate_viability(ind) for ind in population]
    for g in range(GENS):
        print(f"\n-- Gen {g+1}/{GENS} --")
        # Selection score: curriculum (viability score first, then true fitness)
        if g < CURRICULUM_GENS:
            sel_scores = np.array([e["viab_score"] for e in evals], dtype=np.float64)
        else:
            sel_scores = np.array([e["fitness"] for e in evals], dtype=np.float64)

        # Replace -inf/very bad with tiny to keep tournaments working
        if not np.isfinite(sel_scores).all():
            sel_scores[~np.isfinite(sel_scores)] = -1e6

        # Elitism
        elite_idx = np.argsort(sel_scores)[::-1][:ELITES]
        new_pop = [population[i] for i in elite_idx]
        new_evals = [evals[i] for i in elite_idx]

        # Fill the rest with children
        while len(new_pop) < POP:
            i, j = parents_tournament(population, sel_scores, TOURNAMENT_K)
            c1, c2 = make_children(population[i], population[j])
            new_pop.append(c1)
            if len(new_pop) < POP:
                new_pop.append(c2)

        # Evaluate
        evals_children = [evaluate_viability(ind) for ind in new_pop[ELITES:]]
        evals = new_evals + evals_children
        population = new_pop

        # Log survivors (top 10)
        score_name = "viab_score" if g < CURRICULUM_GENS else "fitness"
        sc = np.array([e[score_name] for e in evals])
        order = np.argsort(sc)[::-1]
        print(f"Top 10 by {score_name}:")
        for r in order[:10]:
            e = evals[r]
            print(f"  {e[score_name]: .4f} | nu={e['nu']}  rand={e['rand_prog']:.3f}  probe={e['probe_prog']:.3f}")

    # Final selection by true fitness
    final_scores = np.array([e["fitness"] for e in evals])
    best_idx = int(np.argmax(final_scores))
    best_ind = population[best_idx]
    best_eval = evals[best_idx]
    print("\n=== GA complete ===")
    print(f"Best true fitness: {best_eval['fitness']:.4f} | nu={best_eval['nu']} | probe={best_eval['probe_prog']:.3f}")

    # Save JSON + video
    robot_graph = best_eval["robot_graph"]
    save_graph_as_json(robot_graph, OUT / "best_robot.json")
    print(f"Saved best robot graph to {OUT/'best_robot.json'}")

    # Video (same controller as true_fitness)
    mj.set_mjcb_control(None)
    world, model, data = _spawn_world(robot_graph, SPAWN_Z_OFFSET)
    nu = model.nu
    A = 0.30 * CONTROL_BOUND; FREQ = 0.7; RAMP = 0.8
    PHASES = 2*np.pi * np.arange(nu)/max(1,nu)

    def eval_cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME:
            if m.nu: d.ctrl[:] = 0.0; return
        t = d.time - QUIET_TIME
        A_t = A*(1 - np.exp(-RAMP*t))
        u = A_t * np.sin(2*np.pi*FREQ*t + PHASES)
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    mj.set_mjcb_control(eval_cb)

    recorder = VideoRecorder(output_folder=str(OUT / "videos"))
    video_renderer(model, data, duration=FINAL_DUR, video_recorder=recorder)
    print(f"Saved video in {OUT/'videos'}")

if __name__ == "__main__":
    run_outer_ga()
