"""
Evolutionary algorithm to evolve the robot body (A3).
Outer loop (GA) mutates ONLY the three NDE input vectors.
Inner loop (per-body) trains a CPG controller via CMA-ES (EvoTorch).
Keeps original features: saves best robot JSON and renders a video.

Notes:
- Environment: OlympicArena
- Controller: CPG with per-actuator phase_i + global amplitude + global frequency
- Actions are smoothed toward target and clipped to [-pi/2, pi/2] and actuator ranges
- Fitness in the outer loop: negative 3D distance to TARGET_POSITION (maximization)

Dependencies:
    pip install evotorch torch numpy mujoco matplotlib
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any

import os
import math
import numpy as np
import numpy.typing as npt
import torch
import mujoco as mj
import matplotlib.pyplot as plt
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.core import SolutionBatch
from evotorch.logging import StdOutLogger

# === Local libraries (from your template) ===
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# -----------------------
# Global config / seeds
# -----------------------
SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# === EA parameters ===
POP_SIZE = 10
GENS = 5
GENE_LEN = 64
SIM_DURATION = 10.0
MUTATION_STD = 0.1
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.8
SPAWN_POS = [-0.8, 0, 0.1]
NUM_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# === Inner trainer (CMA-ES) hyperparams ===
# Keep these modest so the outer GA remains affordable. Tweak if needed.
CPG_ALPHA = 0.12              # smoothing coefficient toward target
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
AMP_MIN, AMP_MAX = 0.0, 1.0
FREQ_MIN, FREQ_MAX = 0.5, 2.0
CMA_POPSIZE = 32
CMA_GENERATIONS = 20
CMA_SIGMA_INIT = 0.30
CMA_LOG = False  # set True if you want per-body logs

# data directory
CWD = Path.cwd()
OUTPUT = CWD / "__output__"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------
# Original simple (fixed) controller - not used for evaluation
# ------------------------------------
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    """
    Minimal fixed-parameter CPG (global sine) controller.
    (Kept here for reference; not used in evaluation anymore.)
    """
    nu: int = model.nu
    t: float = float(data.time)

    CONTROL_BOUND = np.pi / 2
    A = 0.6 * CONTROL_BOUND
    FREQ_HZ = 0.7
    OMEGA = 2.0 * np.pi * FREQ_HZ
    BIAS = 0.0

    if nu == 0:
        return np.zeros(0, dtype=np.float64)

    phases = 2.0 * np.pi * (np.arange(nu, dtype=np.float64) / nu)
    u = A * np.sin(OMEGA * t + phases) + BIAS
    np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
    return u

# ------------------------------------
# Genotype operations (unchanged)
# ------------------------------------
def random_genotype() -> List[np.ndarray]:
    """Initialize random genotype vectors (3 x 64 in [0,1])."""
    return [
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
    ]

def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-point crossover between two genotype vectors."""
    L = a.size
    point = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1.astype(np.float32), c2.astype(np.float32)

def crossover_per_chromosome(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """From two parents, perform one-point crossover for each of the three vector types."""
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
    """Perform gaussian mutation on a genotype (elementwise with probability)."""
    mutated = []
    for chrom in gen:
        to_mut = RNG.random(chrom.shape) < MUTATION_PROB
        noise = RNG.normal(loc=0.0, scale=MUTATION_STD, size=chrom.shape).astype(np.float32)
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0)
        mutated.append(new_chrom.astype(np.float32))
    return mutated

# ------------------------------------
# Robot decoding & fitness utility
# ------------------------------------
def fitness_function(history: list[float]) -> float:
    """Calculate fitness as negative 3D cartesian distance to TARGET_POSITION (maximize)."""
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance

def decode_and_build(genotype: List[np.ndarray]):
    """Decode NDE genotype and build a robot model core."""
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core

# ------------------------------------
# CPG controller utilities (generalized A2 logic)
# ------------------------------------
def _make_cpg_policy_with_smoothing(
    model: mj.MjModel,
    params: np.ndarray,
    alpha: float = CPG_ALPHA,
) -> callable:
    """
    Returns a closure f(model, data) -> control ndarray
    Implements:
      target_i(t) = center_i + half_span_i * AMP * sin(2pi * FREQ * t + phase_i)
      ctrl <- ctrl + alpha * (target - ctrl)
    Clipped to both actuator ranges and [-pi/2, pi/2].
    """
    nu = int(model.nu)
    assert params.shape[0] == nu + 2, f"Expected {nu+2} params, got {params.shape[0]}"

    phases = np.asarray(params[:nu], dtype=np.float64)
    AMP = float(params[nu])
    FREQ = float(params[nu + 1])

    # Safety clamp to bounds
    phases = np.clip(phases, PHASE_MIN, PHASE_MAX)
    AMP = float(np.clip(AMP, AMP_MIN, AMP_MAX))
    FREQ = float(np.clip(FREQ, FREQ_MIN, FREQ_MAX))

    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()

    center = 0.5 * (hi + lo)
    half_span = 0.5 * (hi - lo)

    # Internal smoothed control state (closure)
    prev_ctrl = np.zeros(nu, dtype=np.float64)
    HARD_BOUND = np.pi / 2  # assignment bound

    def f(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        t = float(d.time)
        y = AMP * np.sin(2.0 * math.pi * FREQ * t + phases)  # [-AMP, +AMP]

        # Rescale to actuator range
        target = center + half_span * np.clip(y, -1.0, 1.0)

        # Smooth toward target
        nonlocal prev_ctrl
        prev_ctrl = prev_ctrl + alpha * (target - prev_ctrl)

        # Hard-clip to actuator range and assignment bound
        np.clip(prev_ctrl, lo, hi, out=prev_ctrl)
        np.clip(prev_ctrl, -HARD_BOUND, HARD_BOUND, out=prev_ctrl)
        return prev_ctrl

    return f

def _find_core_geom_id(model: mj.MjModel) -> int:
    """Find the geom id whose name contains 'core' (case-insensitive)."""
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name.lower():
            return gid
    raise RuntimeError("No geom with 'core' found in this model")

def quick_viability_gate(model: mj.MjModel, duration: float = 0.2) -> bool:
    """
    Brief 0-control rollout to filter obviously unstable bodies.
    Returns True if finite and stable, else False.
    """
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    def zero_cb(m, d):
        if d.ctrl is not None:
            d.ctrl[:] = 0.0
    mj.set_mjcb_control(zero_cb)
    try:
        # quick manual stepping (avoid extra wrappers)
        steps_per_sec = int(round(1.0 / model.opt.timestep))
        for _ in range(int(round(duration * steps_per_sec))):
            mj.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            return False
    except Exception:
        return False
    finally:
        mj.set_mjcb_control(None)
    return True

# ------------------------------------
# Per-body CMA-ES trainer (Baldwinian learning)
# ------------------------------------
class BodyCPGProblem(Problem):
    """
    EvoTorch Problem that evaluates CPG params on a provided MuJoCo model in OlympicArena.
    Objective: minimize distance to TARGET_POSITION after SIM_DURATION seconds.
    """
    def __init__(self, model: mj.MjModel):
        self.model = model
        self.steps_per_sec = int(round(1.0 / self.model.opt.timestep))
        self.core_geom_id = _find_core_geom_id(self.model)

        nu = int(model.nu)
        self.param_len = nu + 2

        lo = np.concatenate([np.full(nu, PHASE_MIN), [AMP_MIN], [FREQ_MIN]]).astype(np.float64)
        hi = np.concatenate([np.full(nu, PHASE_MAX), [AMP_MAX], [FREQ_MAX]]).astype(np.float64)

        super().__init__(
            objective_sense="min",                   # minimize distance
            solution_length=self.param_len,
            dtype=torch.float64,
            device="cpu",
            initial_bounds=(torch.from_numpy(lo), torch.from_numpy(hi)),
        )

    def _distance_after_rollout(self, params: np.ndarray) -> float:
        """
        Roll out with given params, return Euclidean distance to TARGET_POSITION.
        Uses direct MuJoCo stepping and geom_xpos for robustness and speed.
        """
        data = mj.MjData(self.model)
        mj.mj_resetData(self.model, data)

        # Build policy and set control callback
        policy = _make_cpg_policy_with_smoothing(self.model, params, alpha=CPG_ALPHA)
        def control_cb(m, d):
            u = policy(m, d)
            if d.ctrl is not None:
                d.ctrl[:] = u

        mj.set_mjcb_control(control_cb)
        try:
            horizon = int(round(SIM_DURATION * self.steps_per_sec))
            for _ in range(horizon):
                mj.mj_step(self.model, data)
            pos = data.geom_xpos[self.core_geom_id]  # (3,)
            dx = TARGET_POSITION[0] - float(pos[0])
            dy = TARGET_POSITION[1] - float(pos[1])
            dz = TARGET_POSITION[2] - float(pos[2])
            return float(np.sqrt(dx * dx + dy * dy + dz * dz))
        except Exception:
            return float("+inf")
        finally:
            mj.set_mjcb_control(None)

    def evaluate(self, X: SolutionBatch | torch.Tensor) -> torch.Tensor:
        if isinstance(X, SolutionBatch):
            vals = X.access_values()
            fits = []
            for row in vals:
                params = row.detach().cpu().numpy()
                f = self._distance_after_rollout(params)
                fits.append(f)
            fits_t = torch.as_tensor(fits, dtype=vals.dtype, device=vals.device)
            X.set_evals(fits_t)
            return fits_t
        elif isinstance(X, torch.Tensor):
            fits = [self._distance_after_rollout(row.detach().cpu().numpy()) for row in X]
            return torch.as_tensor(fits, dtype=X.dtype, device=X.device)
        else:
            raise TypeError(f"Unsupported input to evaluate(): {type(X)}")

def train_cpg_controller_for_body(model: mj.MjModel) -> Dict[str, Any]:
    """
    Train CPG params on the given model with CMA-ES.
    Returns dict with best params and best (min) distance.
    """
    if not quick_viability_gate(model):
        return {"best_params": None, "best_fit": float("+inf"), "log": {}}

    prob = BodyCPGProblem(model)
    center = np.zeros(prob.param_len, dtype=np.float64)
    # sensible center: all phases 0, AMP=0.5, FREQ=1.0
    if prob.param_len >= 2:
        center[-2] = 0.5
        center[-1] = 1.0

    solver = CMAES(
        prob,
        popsize=CMA_POPSIZE,
        stdev_init=CMA_SIGMA_INIT,
        center_init=torch.from_numpy(center),
    )

    if CMA_LOG:
        _ = StdOutLogger(solver, interval=1)

    best_params = None
    best_fit = float("+inf")

    for _ in range(CMA_GENERATIONS):
        solver.step()
        pop = solver.population
        vals = pop.access_values().detach().cpu().numpy()
        fits_t = pop.get_evals() if hasattr(pop, "get_evals") else pop.evals
        fits = fits_t.detach().cpu().numpy()

        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit = float(fits[i])
            best_params = vals[i].copy()

    # Fallback if needed
    if best_params is None:
        pop = solver.population
        vals = pop.access_values().detach().cpu().numpy()
        fits_t = pop.get_evals() if hasattr(pop, "get_evals") else pop.evals
        fits = fits_t.detach().cpu().numpy()
        i = int(np.argmin(fits))
        best_fit = float(fits[i])
        best_params = vals[i].copy()

    return {"best_params": best_params, "best_fit": best_fit, "log": {}}

# ------------------------------------
# Genotype evaluation (now uses inner trainer)
# ------------------------------------
def evaluate_genotype(genotype: List[np.ndarray]) -> Tuple[float, list, Dict[str, Any]]:
    """
    Build body from genotype, train CPG controller for this body via CMA-ES,
    then roll out once with the best params to compute fitness and history.

    Returns:
        fitness (float), hist (list of xyz), aux (dict)
        aux contains "best_params" and "robot_graph"
    """
    robot_graph, core = decode_and_build(genotype)

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Train controller for THIS body
    train_res = train_cpg_controller_for_body(model)
    best_params = train_res.get("best_params", None)

    if best_params is None or not np.isfinite(train_res.get("best_fit", np.inf)):
        # Body not viable or training failed → heavy penalty
        return -1e6, [], {"best_params": None, "robot_graph": robot_graph}

    # One rollout for scoring + history (use Tracker only here, with a valid world.spec)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    policy = _make_cpg_policy_with_smoothing(model, best_params, alpha=CPG_ALPHA)

    def control_cb(m, d):
        u = policy(m, d)
        if d.ctrl is not None:
            d.ctrl[:] = u

    mj.set_mjcb_control(control_cb)

    simple_runner(model, data, duration=SIM_DURATION, steps_per_loop=100)

    xpos_history = tracker.history.get("xpos", {})
    if len(xpos_history) == 0 or 0 not in xpos_history:
        return -1e6, [], {"best_params": best_params, "robot_graph": robot_graph}

    hist = xpos_history[0]
    fit = fitness_function(hist)
    return fit, hist, {"best_params": best_params, "robot_graph": robot_graph}

# ------------------------------------
# EA loop (kept; now stores per-body controllers too)
# ------------------------------------
def run_ea():
    mj.set_mjcb_control(None)
    population = [random_genotype() for _ in range(POP_SIZE)]
    fitnesses: List[float] = []
    histories: List[list] = []
    controllers: List[np.ndarray | None] = []
    graphs: List[Any] = []

    # Initial evaluation
    for i, gen in enumerate(population):
        fit, his, aux = evaluate_genotype(gen)
        fitnesses.append(fit)
        histories.append(his)
        controllers.append(aux.get("best_params"))
        graphs.append(aux.get("robot_graph"))
        print(f"Init {i+1}/{POP_SIZE}: fitness={fit:.4f}")

    for gen_idx in range(GENS):
        print(f"\n=== Generation {gen_idx+1} ===")

        # Roulette selection probabilities
        f_arr = np.array(fitnesses)
        weights = f_arr - f_arr.min() + 1e-6
        probs = weights / weights.sum()

        # Create children
        children: List[List[np.ndarray]] = []
        while len(children) < POP_SIZE:
            pa, pb = RNG.choice(len(population), size=2, replace=False, p=probs)
            c1, c2 = crossover_per_chromosome(population[pa], population[pb])
            children.append(gaussian_mutation(c1))
            if len(children) < POP_SIZE:
                children.append(gaussian_mutation(c2))

        # Evaluate children
        child_fitnesses: List[float] = []
        child_histories: List[list] = []
        child_controllers: List[np.ndarray | None] = []
        child_graphs: List[Any] = []

        for i, chi in enumerate(children):
            fit, his, aux = evaluate_genotype(chi)
            child_fitnesses.append(fit)
            child_histories.append(his)
            child_controllers.append(aux.get("best_params"))
            child_graphs.append(aux.get("robot_graph"))
            print(f"Child {i+1}/{len(children)}: fitness={fit:.4f}")

        # Elitist survivor selection
        combined = population + children
        combined_fit = fitnesses + child_fitnesses
        combined_hist = histories + child_histories
        combined_ctrl = controllers + child_controllers
        combined_graph = graphs + child_graphs

        order = np.argsort(combined_fit)[::-1][:POP_SIZE]
        population = [combined[i] for i in order]
        fitnesses = [combined_fit[i] for i in order]
        histories = [combined_hist[i] for i in order]
        controllers = [combined_ctrl[i] for i in order]
        graphs = [combined_graph[i] for i in order]

        print("Survivors:")
        for r, f in enumerate(fitnesses):
            print(f"{r+1}: {f:.4f}")

    # Final best
    best_idx = int(np.argmax(fitnesses))
    best_gen = population[best_idx]
    best_hist = histories[best_idx]
    best_fit = fitnesses[best_idx]
    best_params = controllers[best_idx]
    best_graph = graphs[best_idx]

    print("\n=== EA finished ===")
    print(f"Best fitness: {best_fit:.4f}")

    # Save best robot JSON
    save_graph_as_json(best_graph, OUTPUT / "best_robot.json")
    print(f"Saved best robot graph to {OUTPUT/'best_robot.json'}")

    # Render video of best robot with its learned controller
    mj.set_mjcb_control(None)
    _, core = decode_and_build(best_gen)  # rebuild to ensure consistency
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    if best_params is None:
        # fallback to fixed controller if something went wrong
        ctrl_policy = nn_controller
    else:
        policy = _make_cpg_policy_with_smoothing(model, best_params, alpha=CPG_ALPHA)
        def ctrl_policy(m, d):
            u = policy(m, d)
            if d.ctrl is not None:
                d.ctrl[:] = u

    ctrl = Controller(controller_callback_function=ctrl_policy, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    video_folder = OUTPUT / "videos"
    video_folder.mkdir(exist_ok=True)
    recorder = VideoRecorder(output_folder=str(video_folder))
    video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    print(f"Saved video of best robot to {video_folder}")

    return best_gen, best_hist, best_fit


if __name__ == "__main__":
    run_ea()
