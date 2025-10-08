"""
Assignment 3 — Co-evolution with EvoTorch (outer) + Nevergrad CMA (inner CPG)

WHAT'S NEW (vs your A3 template):
- Outer loop: EvoTorch GA evolves ONLY the three NDE input vectors (3 x 64, in [0,1]).
- Inner loop: Per-body controller training via Nevergrad CMA-ES on an enhanced CPG parametrization
  (per-actuator phases + per-actuator amplitude + per-actuator bias + global frequency),
  actions clipped to ±π/2.
- Single fitness ONLY: negative distance to TARGET_POSITION after rollout.
- Cheap viability gates (do not change the objective): actuator band check, and 0.2s zero-control sanity.
- Caching: same genotype → reuse best learned fitness and controller params.
- GA: boosted exploration (uniform crossover + mutation + random immigrants).
- NDE/HPD are instantiated ONCE for the whole run so genotype→phenotype mapping is stable.
- Pure pruning: we never edit graphs; unbuildable bodies get a very poor fitness.
- Saving: reuse the cached graph that actually built during evolution.
- End: saves best robot JSON at __output__/robot_graph.json and a video in __output__/videos/.

Dependencies:
  pip install evotorch nevergrad
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party
import numpy as np
import numpy.typing as npt
import mujoco as mj
import inspect
import random
import copy
import hashlib

# EvoTorch outer EA
from evotorch import Problem, Solution
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import Operator
from evotorch.core import SolutionBatch

# torch
import torch

# Inner-loop optimizer (CMA-ES)
import nevergrad as ng

# --- ARIEL imports (unchanged) ---
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

# ========= Config =========
SEED = 42
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# World / task
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = np.array([5.0, 0.0, 0.5], dtype=np.float64)
NUM_OF_MODULES = 30

# Genotype (body) = three 64-dim vectors in [0,1]
GENE_LEN = 64
CHROMOSOMES = 3
BOUNDS = (0.0, 1.0)

# Outer GA  << exploration (tune up when stable)
POP_SIZE = 10            # try 50–100 when sims are stable/time allows
NGENS = 50
TOURNAMENT_K = 2
CROSSOVER_RATE = 0.8     # we also add an explicit uniform crossover operator
MUT_P = 0.30
MUT_SIGMA = 0.25

# Viability gates (no change to objective)
NU_MIN, NU_MAX = 2, 20          # <-- widened lower bound from 4→2 to avoid pruning borderline bodies
ZERO_STAB_SEC = 0.2

# Simulation
CONTROL_BOUND = np.pi / 2
QUIET_TIME = 0.60
SIM_DURATION = 10.0     # final rollout (fitness and video)
INNER_SIM = 6.0         # inner-loop rollout per CMA candidate (shorter is faster)

# Inner CMA-ES budgets (per body)
CMA_BUDGET_EVALS = 20   # try 60–120 when stable
CMA_WORKERS = 1         # MuJoCo safer per-process

# I/O
CWD = Path.cwd()
OUT = CWD / "__output__"
(OUT / "videos").mkdir(parents=True, exist_ok=True)

# ========= Helpers =========
def fitness_function(history: List[Tuple[float, float, float]]) -> float:
    """Negative Euclidean distance to TARGET_POSITION."""
    if not history:
        return -1e6
    xc, yc, zc = history[-1]
    diff = TARGET_POSITION - np.array([xc, yc, zc], dtype=np.float64)
    return -float(np.linalg.norm(diff))

def extract_end_xyz_or_fallback(model: mj.MjModel, data: mj.MjData) -> Tuple[float, float, float]:
    """If tracker is empty, read 'core' body pos or fallback to first non-world body."""
    try:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "core")
        pos = np.array(data.xpos[bid], dtype=np.float64)
    except Exception:
        bid = 1 if model.nbody > 1 else 0
        pos = np.array(data.xpos[bid], dtype=np.float64)
    if not np.isfinite(pos).all():
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return float(pos[0]), float(pos[1]), float(pos[2])

def zero_control_stability(model: mj.MjModel, duration: float = ZERO_STAB_SEC) -> bool:
    """Very cheap sanity check; returns False if the sim explodes (NaNs)."""
    mj.set_mjcb_control(None)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    def cb(m: mj.MjModel, d: mj.MjData):
        if m.nu:
            d.ctrl[:] = 0.0
    mj.set_mjcb_control(cb)
    try:
        simple_runner(model, data, duration=duration, steps_per_loop=40)
    except Exception:
        return False
    return np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()

# ---- Build helpers (pure pruning; no graph edits) ----
def try_build_from_graph(robot_graph):
    """Attempt to build MuJoCo spec from an existing graph; return (core, world, model) or None."""
    sx, sy = float(SPAWN_POS[0]), float(SPAWN_POS[1])
    # More generous spawn heights
    candidate_z = [max(float(SPAWN_POS[2]), 0.20), 0.30, 0.40, 0.60, 0.80, 1.00, 1.20]
    try:
        mj.set_mjcb_control(None)
    except Exception:
        pass
    try:
        core = construct_mjspec_from_graph(robot_graph)
    except Exception:
        return None
    for sz in candidate_z:
        try:
            world = OlympicArena()
            # Allow the environment helper to correct the spawn for bounding boxes
            world.spawn(core.spec, spawn_position=[sx, sy, float(sz)], correct_for_bounding_box=True)
            model = world.spec.compile()
            return core, world, model
        except Exception:
            continue
    return None

def nde_decode_build(
    genotype: List[np.ndarray],
    nde: NeuralDevelopmentalEncoding,
    hpd: HighProbabilityDecoder,
):
    """Deterministic (per-genotype) decode + build with an expanded, fixed resample ladder."""
    # Stable seed from SHA-256 of the flat genotype bytes (NOT Python hash)
    flat = np.concatenate([genotype[0].ravel(), genotype[1].ravel(), genotype[2].ravel()]).astype(np.float32)
    digest = hashlib.sha256(flat.tobytes()).digest()
    base_seed = int.from_bytes(digest[:4], "little")  # 32-bit

    # Save RNG states so we don't affect global randomness
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        with torch.no_grad():
            p_type, p_conn, p_rot = nde.forward(genotype)

        # Try more deterministic “neighbor” seeds to get a buildable sample (8 -> 64)
        for offset in range(64):  # 0..63
            seed = (base_seed + offset) & 0xFFFFFFFF
            np.random.seed(seed)
            random.seed(seed)

            robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
            built = try_build_from_graph(robot_graph)
            if built is not None:
                core, world, model = built
                return robot_graph, core, world, model

        # If none of the 64 decodes built, prune it
        raise RuntimeError("Body not buildable from decoded graph (after 64 deterministic attempts)")

    finally:
        # Restore RNG state
        np.random.set_state(np_state)
        random.setstate(py_state)

# ========= Enhanced CPG =========
def make_cpg_callback(model: mj.MjModel, phases: np.ndarray, amps: np.ndarray, bias: np.ndarray, freq_hz: float):
    """Per-joint sine CPG with per-joint amplitude and bias; global frequency."""
    nu = int(model.nu)
    phases = np.asarray(phases, dtype=np.float64)
    amps = np.asarray(amps, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    f = float(np.clip(freq_hz, 0.3, 2.0))
    omega = 2.0 * np.pi * f
    def cb(m: mj.MjModel, d: mj.MjData):
        if d.time < QUIET_TIME:
            if m.nu:
                d.ctrl[:] = 0.0
            return
        t = d.time - QUIET_TIME
        u = amps * np.sin(omega * t + phases) + bias
        np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
        d.ctrl[:] = u
    return cb

def clamp_theta_cpg2(nu: int, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """theta = [phases(nu), amps(nu), bias(nu), freq]."""
    theta = np.asarray(theta, dtype=np.float64)
    phases = np.mod(theta[:nu], 2.0 * np.pi)
    amps = np.clip(theta[nu:2 * nu], 0.0, CONTROL_BOUND)
    bias = np.clip(theta[2 * nu:3 * nu], -0.25 * np.pi, 0.25 * np.pi)
    f = float(np.clip(theta[-1], 0.3, 2.0))
    return phases, amps, bias, f

# ========= Inner loop: CMA-ES on CPG (Nevergrad) =========
def optimize_cpg_cma(spec, model: mj.MjModel, sim_seconds: float = INNER_SIM,
                     budget_evals: int = CMA_BUDGET_EVALS) -> Tuple[float, Dict[str, Any]]:
    """Train a CPG for THIS body using Nevergrad CMA; returns (best_fitness, ctrl_params)."""
    nu = int(model.nu)
    if nu == 0:
        return -1e6, {"phases": np.zeros(0), "amps": np.zeros(0), "bias": np.zeros(0), "f": 0.8}

    # [phases(nu)∈[0,2π), amps(nu)∈[0,π/2], bias(nu)∈[-π/4,π/4], freq∈[0.3,2.0]]
    lower = np.concatenate([np.zeros(nu), np.zeros(nu), -0.25 * np.pi * np.ones(nu), [0.3]])
    upper = np.concatenate([2.0 * np.pi * np.ones(nu), CONTROL_BOUND * np.ones(nu), 0.25 * np.pi * np.ones(nu), [2.0]])
    init = np.concatenate([
        RNG.uniform(0.0, 2.0 * np.pi, size=nu),
        0.30 * CONTROL_BOUND * np.ones(nu),
        RNG.uniform(-0.05 * np.pi, 0.05 * np.pi, size=nu),
        [0.8],
    ])

    instrumentation = ng.p.Array(init=init).set_bounds(lower, upper)
    # Smaller sigma to avoid NG warning; roughly 1/6 of smallest range
    instrumentation = instrumentation.set_mutation(sigma=0.25)

    opt = ng.optimizers.CMA(parametrization=instrumentation, budget=budget_evals, num_workers=CMA_WORKERS)

    def evaluate_theta(th: np.ndarray) -> float:
        phases, amps, bias, f = clamp_theta_cpg2(nu, th)
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(spec, data)
        mj.set_mjcb_control(None)
        mj.set_mjcb_control(make_cpg_callback(model, phases, amps, bias, f))
        try:
            simple_runner(model, data, duration=sim_seconds, steps_per_loop=80)
        except Exception:
            return 1e6  # NG minimizes
        xpos = tracker.history.get("xpos", {}).get(0, [])
        if not xpos:
            xpos = [extract_end_xyz_or_fallback(model, data)]
        return -fitness_function(xpos)  # loss

    for _ in range(budget_evals):
        cand = opt.ask()
        loss = evaluate_theta(cand.value)
        opt.tell(cand, loss)

    rec = opt.provide_recommendation()
    phases, amps, bias, f = clamp_theta_cpg2(nu, rec.value)

    # Re-eval best as positive fitness
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(spec, data)
    mj.set_mjcb_control(None)
    mj.set_mjcb_control(make_cpg_callback(model, phases, amps, bias, f))
    try:
        simple_runner(model, data, duration=sim_seconds, steps_per_loop=80)
    except Exception:
        return -1e6, {"phases": phases, "amps": amps, "bias": bias, "f": f}
    xpos = tracker.history.get("xpos", {}).get(0, [])
    if not xpos:
        xpos = [extract_end_xyz_or_fallback(model, data)]
    return fitness_function(xpos), {"phases": phases, "amps": amps, "bias": bias, "f": f}

# ========= EvoTorch pieces: operators + Problem =========
class GaussianMut(Operator):
    """Per-gene Gaussian mutation with clipping to [0,1], in-place on a SolutionBatch."""
    def __init__(self, problem, mut_p=MUT_P, sigma=MUT_SIGMA):
        super().__init__(problem)
        self.mut_p = float(mut_p)
        self.sigma = float(sigma)

    def _do(self, batch):  # mutate in place
        vals = batch.access_values(keep_evals=False)  # shape: [n, L], torch.float32
        n, L = vals.shape
        assert L == CHROMOSOMES * GENE_LEN, f"Got genome len {L}, expected {CHROMOSOMES*GENE_LEN}"
        x = vals.view(n, CHROMOSOMES, GENE_LEN)       # [n, 3, 64]
        mask = (torch.rand_like(x) < self.mut_p)      # bool mask
        noise = torch.randn_like(x) * self.sigma
        x.add_(noise * mask)                          # in-place add
        x.clamp_(0.0, 1.0)                            # in-place clip

class UniformCrossover(Operator):
    """Pairwise uniform crossover over flat genome [0,1]."""
    def __init__(self, problem, rate: float = 0.5):
        super().__init__(problem)
        self.rate = float(rate)

    def _do(self, batch: SolutionBatch):
        vals = batch.access_values(keep_evals=False)   # [n, L]
        n, L = vals.shape
        if n < 2:
            return
        mates = vals[torch.randperm(n)]
        mask = (torch.rand(n, L, device=vals.device) < self.rate)
        vals.copy_(torch.where(mask, mates, vals))

def _hash_genotype(arr3x64: np.ndarray) -> bytes:
    return arr3x64.astype(np.float32).tobytes()

class BodyProblem(Problem):
    """Outer-loop fitness = best learned controller fitness per body (single objective)."""
    def __init__(self):
        super().__init__(
            solution_length=CHROMOSOMES * GENE_LEN,
            bounds=BOUNDS,
            objective_sense="max",
        )
        # Persistent decoder stack: one NDE/HPD per run
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        self.nde.eval()
        for p in self.nde.parameters():
            p.requires_grad_(False)
        self.hpd = HighProbabilityDecoder(NUM_OF_MODULES)

        self.cache: Dict[bytes, Dict[str, Any]] = {}
        # Telemetry
        self.stats = {"evals": 0, "built": 0, "bad_nu": 0, "unstable": 0}

    def _eval_one(self, solution: Solution):
        self.stats["evals"] += 1

        arr = solution.values.detach().cpu().numpy().astype(np.float32)
        arr = arr.reshape(CHROMOSOMES, GENE_LEN)  # (3, 64)

        key = _hash_genotype(arr)
        if key in self.cache:
            rec = self.cache[key]
            solution.set_evaluation(rec["fitness"])
            return

        genotype = [arr[0], arr[1], arr[2]]

        # Decode & build with persistent NDE/HPD (pure pruning + deterministic resampling)
        try:
            robot_graph, core, world, model = nde_decode_build(genotype, self.nde, self.hpd)
            self.stats["built"] += 1
        except Exception:
            solution.set_evaluation(-1e6)
            return

        nu = int(model.nu)
        if not (NU_MIN <= nu <= NU_MAX):
            self.stats["bad_nu"] += 1
            solution.set_evaluation(-1e6); return
        if not zero_control_stability(model, ZERO_STAB_SEC):
            self.stats["unstable"] += 1
            solution.set_evaluation(-1e6); return

        # Inner loop: Nevergrad CMA to train a CPG
        best_fit, ctrl = optimize_cpg_cma(world.spec, model, sim_seconds=INNER_SIM, budget_evals=CMA_BUDGET_EVALS)

        self.cache[key] = {"fitness": best_fit, "nu": nu, "ctrl": ctrl, "graph": copy.deepcopy(robot_graph)}
        solution.set_evaluation(best_fit)

    def evaluate(self, solutions):
        if isinstance(solutions, SolutionBatch):
            for i in range(len(solutions)):
                self._eval_one(solutions[i])
        elif isinstance(solutions, Solution):
            self._eval_one(solutions)
        else:
            for sol in solutions:
                self._eval_one(sol)

class GAWithImmigrants(GeneticAlgorithm):
    """Standard GA step + inject K random immigrants each generation."""
    def __init__(self, problem, *, immigrants_frac: float = 0.10, **kwargs):
        super().__init__(problem, **kwargs)
        self._imm_frac = float(immigrants_frac)

    def _step(self):
        # 1) Do the standard GA step (reproduction + elitist selection)
        super()._step()

        # 2) Add immigrants and reselect
        pop = self._population                      # SolutionBatch
        P = len(pop)
        K = max(1, int(P * self._imm_frac))

        # Fresh random solutions
        rnd = self._problem.generate_batch(K)
        self._problem.evaluate(rnd)                 # make sure they have evals

        # Concatenate and keep the best P individuals
        extended = SolutionBatch.cat([pop, rnd])    # new SolutionBatch
        self._population = extended.take_best(P)    # respects objective_sense

# ========= Run: GA → save best JSON + video =========
def run():
    try:
        mj.set_mjcb_control(None)
    except Exception:
        pass

    problem = BodyProblem()

    ga_kwargs = dict(
        popsize=POP_SIZE,
        operators=[UniformCrossover(problem, rate=0.5), GaussianMut(problem)],
        elitist=True,
    )

    sig = inspect.signature(GeneticAlgorithm)
    if "tournament_size" in sig.parameters:
        ga_kwargs["tournament_size"] = TOURNAMENT_K
    if "crossover_rate" in sig.parameters:
        ga_kwargs["crossover_rate"] = 1.0  # we also apply an explicit crossover operator

    ga = GAWithImmigrants(problem, immigrants_frac=0.10, **ga_kwargs)

    ga.run(num_generations=NGENS)

    # Telemetry
    print("Evals: {evals} | built: {built} | bad_nu: {bad_nu} | unstable: {unstable}".format(**problem.stats))

    best: Solution = ga.status["pop_best"]
    best_arr = best.values.detach().cpu().numpy().astype(np.float32).reshape(CHROMOSOMES, GENE_LEN)
    best_gen = [best_arr[0], best_arr[1], best_arr[2]]

    # Prefer the cached graph that actually built during evolution
    key = _hash_genotype(np.asarray(best_arr, dtype=np.float32))
    cached = problem.cache.get(key)

    if cached is not None:
        robot_graph = cached["graph"]
        built = try_build_from_graph(robot_graph)
        if built is None:
            # Fallback to re-decode (deterministic with persistent NDE/HPD + resample ladder)
            robot_graph, core, world, model = nde_decode_build(best_gen, problem.nde, problem.hpd)
        else:
            core, world, model = built
    else:
        robot_graph, core, world, model = nde_decode_build(best_gen, problem.nde, problem.hpd)

    out_json = OUT / "robot_graph.json"
    save_graph_as_json(robot_graph, out_json)
    print(f"Saved best robot graph to {out_json}")

    # Train controller again for video (or fetch from cache if you prefer)
    best_fit, ctrl = optimize_cpg_cma(world.spec, model, sim_seconds=INNER_SIM, budget_evals=CMA_BUDGET_EVALS)
    print(f"Best fitness (re-eval): {best_fit:.4f}")

    # Final video over SIM_DURATION with learned controller
    mj.set_mjcb_control(None)
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    nu = int(model.nu)
    phases = ctrl.get("phases", np.zeros(nu))
    amps = ctrl.get("amps", 0.30 * CONTROL_BOUND * np.ones(nu))
    bias = ctrl.get("bias", np.zeros(nu))
    f = ctrl.get("f", 0.8)

    mj.set_mjcb_control(make_cpg_callback(model, phases, amps, bias, f))
    recorder = VideoRecorder(output_folder=str(OUT / "videos"))
    video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    print(f"Saved video to {OUT/'videos'}")

if __name__ == "__main__":
    run()
