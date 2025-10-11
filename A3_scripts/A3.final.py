"""Assignment 3 – Minimal GA + per-body CMA-ES NA-CPG + export of best robot video and JSON.

This file keeps the teammate’s outer EA (population, crossover, mutation, elitism, selection,
staged sim times, CSV, videos, JSON) EXACTLY the same, and replaces the inner-loop controller
with the Body-Agnostic NA-CPG (one oscillator per actuator) trained by CMA-ES per body.

Inner-loop CMA-ES optimizes theta = [phase_0..phase_{nu-1}, AMP, FREQ].
NA-CPG parameters & behavior follow the user’s spec:
- Controller smoothing inside Controller: alpha = 0.6
- Internal oscillator alpha ≈ 0.45, COUP = 0.08
- Defaults: amplitudes[:] = 0.9, w[:] = 2π*1.5 Hz, phase[:] = 0.0
- Mapping: oscillator output → joint center ± half-span (respect actuator ctrlrange), then clip
- Bounds on theta: phase ∈ [-π, π], AMP ∈ [0.0, 1.5], FREQ ∈ [0.8, 3.0]
"""

# ---------- Imports ----------
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Dict, Tuple, Optional
import math
import numpy as np
import numpy.typing as npt
import mujoco as mj
import csv
from datetime import datetime

import torch
from torch import nn
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.core import SolutionBatch
from evotorch.logging import StdOutLogger

from ariel import console
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

if TYPE_CHECKING:
    from networkx import DiGraph


# ---------- Globals ----------
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
GENOTYPE_SIZE = 64

# Outer EA loop parameters (evolving the body)  ---- (UNCHANGED)
POP_SIZE = 8
N_GEN = 20
CX_PROB = 0.5
MUT_PROB = 0.3
MUT_SIGMA = 0.3
ELITISM_SIZE = 1
PICK_PARENTS_BETA = 5
if N_GEN > 30:
    SIM_TIME_STAGES = [10.0, 20.0, 40.0, 60.0]
else:
    SIM_TIME_STAGES = [10.0, 10.0, 15.0, 20.0]

# Inner EA loop parameters (per-body CMA-ES)  ---- outer defaults kept, NA-CPG uses its own bounds
MIN_VIABLE_MOVEMENT = 0.015
CPG_TRAINING_POP = 20
CPG_TRAINING_GENS = 15

# Teammate globals (kept; NA-CPG enforces its own):
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
AMP_MIN, AMP_MAX     = 0.0, 1.0
FREQ_MIN, FREQ_MAX   = 0.4, 2.0
SMOOTH_ALPHA         = 0.5  # kept, but NA-CPG replay uses Controller(alpha=0.6)

# NA-CPG controller smoothing (as per user spec)
CTRL_ALPHA = 0.6

# NA-CPG theta bounds (as per user spec)
NA_PHASE_MIN, NA_PHASE_MAX = -math.pi, math.pi
NA_AMP_MIN,   NA_AMP_MAX   = 0.0, 1.5
NA_FREQ_MIN,  NA_FREQ_MAX  = 0.8, 3.0

# Timestamp for output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Dynamic simulation length ----------
def get_sim_time_for_gen(gen: int, total_gens: int = N_GEN) -> float:
    """Return simulation duration based on which quarter of evolution we're in."""
    quarter = total_gens // 4
    if gen < quarter:
        return SIM_TIME_STAGES[0]
    elif gen < 2 * quarter:
        return SIM_TIME_STAGES[1]
    elif gen < 3 * quarter:
        return SIM_TIME_STAGES[2]
    else:
        return SIM_TIME_STAGES[3]


# ---------- Random controller (kept for viability check) ----------
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    input_size = len(data.qpos)
    hidden = 8
    output = model.nu
    if output == 0:
        return np.zeros(0)
    w1 = RNG.normal(0.0, 0.5, size=(input_size, hidden))
    w2 = RNG.normal(0.0, 0.5, size=(hidden, output))
    return np.tanh(np.tanh(data.qpos @ w1) @ w2) * np.pi


# ---------- Fitness ----------
def fitness_function(history: list[list[float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)


# ---------- Simulation ----------
def experiment(robot: Any, controller: Controller, duration: int, record: bool = False) -> None:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    if record:
        video_folder = str(DATA / "videos")
        recorder = VideoRecorder(output_folder=video_folder)
        video_renderer(model, data, duration=duration, video_recorder=recorder)
    else:
        simple_runner(model, data, duration=duration)


# ---------- Genotype operations ----------
def random_genotype() -> list[np.ndarray]:
    """Initialize random genotype vectors"""
    return [
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
    ]


def one_point_crossover(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform one-point crossover between two genotype vectors (of the same type)"""
    L = a.size
    point = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1.astype(np.float32), c2.astype(np.float32)


def crossover_per_chromosome(pa: list[np.ndarray], pb: list[np.ndarray], cx_prob: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """From two parents, perform one-point crossover for each of the three vector types"""
    child1, child2 = [], []
    for idx in range(3):
        if RNG.random() < cx_prob:
            ca, cb = one_point_crossover(pa[idx], pb[idx])
            child1.append(ca)
            child2.append(cb)
        else:
            child1.append(pa[idx].copy())
            child2.append(pb[idx].copy())
    return child1, child2


def gaussian_mutation(gen: list[np.ndarray], mut_prob: float, sigma: float) -> list[np.ndarray]:
    """Perform gaussian mutation on a genotype"""
    mutated = []
    for chrom in gen:
        to_mut = RNG.random(len(chrom)) < mut_prob
        noise = RNG.normal(loc=0.0, scale=sigma, size=len(chrom)).astype(np.float32)
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0)
        mutated.append(new_chrom.astype(np.float32))
    return mutated


def _find_core_geom_id(model: mj.MjModel) -> int | None:
    # Best-effort: find a geom with 'core' in its name for tracking last xyz
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name.lower():
            return gid
    return None


# ---------- NA-CPG (Body-Agnostic) ----------
def _fully_connected_adj(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


class BodyAgnosticNACPG(nn.Module):
    """
    One oscillator per actuator. Oscillator outputs mapped to each joint's center ± half-span.
    Internal coupling COUP = 0.08. Radial gain alpha ≈ 0.45 by default.
    """
    def __init__(
        self,
        adjacency: Dict[int, List[int]],
        alpha: float = 0.45,
        dt: float = 0.01,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n = len(adjacency)
        self.adjacency = adjacency
        self.alpha = float(alpha)
        self.dt = float(dt)

        # Evolvable params (set per rollout)
        self.phase = nn.Parameter(torch.zeros(self.n), requires_grad=False)
        self.amplitudes = nn.Parameter(torch.full((self.n,), 0.9), requires_grad=False)
        self.w = nn.Parameter(torch.full((self.n,), 2.0 * math.pi * 1.5), requires_grad=False)  # ~1.5 Hz

        # Buffers for oscillator state (x,y) per joint
        self.register_buffer("xy", torch.randn(self.n, 2) * 0.05)
        self.register_buffer("xy_dot_old", torch.zeros(self.n, 2))

        # Actuator mapping (bound on first bind)
        self._ctrl_lo: Optional[np.ndarray] = None
        self._ctrl_hi: Optional[np.ndarray] = None

    @classmethod
    def from_model(cls, model: mj.MjModel, *, alpha: float = 0.45, seed: Optional[int] = None) -> "BodyAgnosticNACPG":
        nu = int(model.nu)
        if nu <= 0:
            raise ValueError("Model has zero actuators.")
        inst = cls(adjacency=_fully_connected_adj(nu), alpha=alpha, dt=float(model.opt.timestep), seed=seed)
        inst._bind_ranges(model)
        return inst

    def _bind_ranges(self, model: mj.MjModel) -> None:
        self._ctrl_lo = model.actuator_ctrlrange[:, 0].astype(np.float64)
        self._ctrl_hi = model.actuator_ctrlrange[:, 1].astype(np.float64)

    def step(self) -> np.ndarray:
        """Advance oscillators one MuJoCo step and return normalized outputs in [-1,1] per joint."""
        n = self.n
        xy = self.xy
        xyd = self.xy_dot_old

        # Rotation matrices for phase coupling
        r = torch.zeros(n, n, 2, 2, dtype=xy.dtype)
        I = torch.eye(2, dtype=xy.dtype)
        for i in range(n):
            for j in range(n):
                if i == j:
                    r[i, j] = I
                else:
                    d = self.phase[i] - self.phase[j]
                    c, s = torch.cos(d), torch.sin(d)
                    r[i, j, 0, 0] = c
                    r[i, j, 0, 1] = -s
                    r[i, j, 1, 0] = s
                    r[i, j, 1, 1] = c

        COUP = 0.08
        new_xy = torch.empty_like(xy)
        new_xyd = torch.empty_like(xyd)

        for i in range(n):
            xi, yi = xy[i]
            xdot_old, ydot_old = xyd[i]
            r2 = xi * xi + yi * yi
            a = self.alpha * (1.0 - r2)
            b = self.w[i]

            # local dynamics (Hopf-like)
            local_xdot = a * xi - b * yi
            local_ydot = b * xi + a * yi

            # coupling term (average of neighbors)
            coup = torch.zeros(2, dtype=xy.dtype)
            nbrs = self.adjacency[i]
            if len(nbrs) > 0:
                for j in nbrs:
                    coup += COUP * torch.mv(r[i, j], xy[j])
                coup /= float(len(nbrs))

            xdot = local_xdot + coup[0]
            ydot = local_ydot + coup[1]

            # mild rate limiting
            diff = 10.0
            xdot = torch.clamp(xdot, xdot_old - diff, xdot_old + diff)
            ydot = torch.clamp(ydot, ydot_old - diff, ydot_old + diff)

            new_x = xi + self.dt * xdot
            new_y = yi + self.dt * ydot

            new_xy[i, 0] = new_x
            new_xy[i, 1] = new_y
            new_xyd[i, 0] = xdot
            new_xyd[i, 1] = ydot

        self.xy = new_xy
        self.xy_dot_old = new_xyd

        # Use y as the oscillator output; scale by amplitudes, then normalize to [-1,1] by π/2 mapping at Controller
        out = (self.amplitudes * self.xy[:, 1]).detach().cpu().numpy()
        # normalize to [-1,1] by dividing by (π/2) when mapping to ctrl
        return np.clip(out / (np.pi / 2.0), -1.0, 1.0)

    def control_callback(self, model: mj.MjModel) -> Any:
        if self._ctrl_lo is None:
            self._bind_ranges(model)
        lo, hi = self._ctrl_lo, self._ctrl_hi
        center = 0.5 * (hi + lo)
        half = 0.5 * (hi - lo)

        def _cb(_m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            # advance NA-CPG one step and return target controls (Controller will smooth with alpha=0.6)
            y_norm = self.step()  # [-1,1]
            target = center + half * y_norm
            return np.clip(target, lo, hi)

        return _cb


# ---------- CMA-ES Problem using NA-CPG ----------
class BodyCPGProblem(Problem):
    """EvoTorch problem wrapping a *given compiled model* for a decoded body, using NA-CPG."""
    def __init__(self, model: mj.MjModel, sim_seconds: float = 6.0):
        self.model = model
        self.sim_seconds = float(sim_seconds)
        self.steps_per_sec = int(round(1.0 / model.opt.timestep))
        self.core_gid = _find_core_geom_id(model)

        nu = int(model.nu)
        L = (nu + 2) if nu > 0 else 2  # AMP,FREQ even if no actuators (won't be used)

        # NA-CPG bounds
        lo = np.concatenate([np.full(nu, NA_PHASE_MIN), [NA_AMP_MIN], [NA_FREQ_MIN]]).astype(np.float64)
        hi = np.concatenate([np.full(nu, NA_PHASE_MAX), [NA_AMP_MAX], [NA_FREQ_MAX]]).astype(np.float64)

        super().__init__(
            objective_sense="min",          # minimize final 3D distance
            solution_length=L,
            dtype=torch.float64,
            device="cpu",
            initial_bounds=(torch.from_numpy(lo), torch.from_numpy(hi)),
        )

    def _rollout_distance(self, theta: np.ndarray) -> float:
        # Fresh data for each rollout
        data = mj.MjData(self.model)
        mj.mj_resetData(self.model, data)
        if data.ctrl is not None:
            data.ctrl[:] = 0.0

        nu = int(self.model.nu)
        if nu == 0:
            return 1e6  # no actuators → terrible distance

        # Clamp theta and assign NA-CPG params
        phases = np.clip(theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX)
        AMP  = float(np.clip(theta[nu],     NA_AMP_MIN,  NA_AMP_MAX))
        FREQ = float(np.clip(theta[nu + 1], NA_FREQ_MIN, NA_FREQ_MAX))
        omega = 2.0 * math.pi * FREQ

        # Build NA-CPG
        cpg = BodyAgnosticNACPG.from_model(self.model, alpha=0.45, seed=SEED)
        with torch.inference_mode():
            cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
            cpg.amplitudes[:] = AMP
            cpg.w[:] = omega

        # RAW MuJoCo control callback (no Controller, no tracker)
        # Controller smoothing (alpha=0.6) is applied only in replay/video runs.
        raw_cb = cpg.control_callback(self.model)

        def mjcb(_m: mj.MjModel, d: mj.MjData):
            # Write targets directly into d.ctrl (what Controller would return)
            d.ctrl[:] = raw_cb(_m, d)

        mj.set_mjcb_control(mjcb)

        horizon = int(round(self.sim_seconds * self.steps_per_sec))
        xyz_last = None
        try:
            for _ in range(horizon):
                mj.mj_step(self.model, data)
                if self.core_gid is not None:
                    xyz_last = data.geom_xpos[self.core_gid].copy()
                else:
                    # body root (qpos first 3) as fallback
                    xyz_last = np.array([
                        float(data.qpos[0]),
                        float(data.qpos[1]),
                        float(data.qpos[2] if self.model.nq >= 3 else 0.0),
                    ])
        finally:
            mj.set_mjcb_control(None)

        if xyz_last is None:
            return 1e6

        xt, yt, zt = TARGET_POSITION
        dx, dy, dz = xt - xyz_last[0], yt - xyz_last[1], zt - xyz_last[2]
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))


    def evaluate(self, X):
        if isinstance(X, SolutionBatch):
            vals = X.access_values()
            out = []
            for row in vals:
                theta = row.detach().cpu().numpy()
                d = self._rollout_distance(theta)
                out.append(d)
            outs = torch.as_tensor(out, dtype=vals.dtype, device=vals.device)
            X.set_evals(outs)
            return outs
        elif isinstance(X, torch.Tensor):
            out = [self._rollout_distance(row.detach().cpu().numpy()) for row in X]
            return torch.as_tensor(out, dtype=X.dtype, device=X.device)
        else:
            raise TypeError(f"Unsupported input to evaluate(): {type(X)}")


def optimize_cpg_cma_for_body(model: mj.MjModel, seconds: float = 6.0, pop: int = 40, gens: int = 20) -> np.ndarray:
    """Run CMA-ES on the NA-CPG params for THIS body; return best params (numpy)."""
    nu = int(model.nu)
    if nu == 0:
        return np.array([0.5, 1.0], dtype=np.float64)  # AMP, FREQ (unused)

    prob = BodyCPGProblem(model, sim_seconds=seconds)
    # safe center: phases=0, AMP=0.8 (within [0,1.5]), FREQ=1.5 (within [0.8,3.0])
    center = np.concatenate([np.zeros(nu), [0.8], [1.5]]).astype(np.float64)

    solver = CMAES(
        prob,
        popsize=max(10, int(pop)),
        stdev_init=0.3,
        center_init=torch.from_numpy(center),
    )
    _ = StdOutLogger(solver, interval=max(1, gens // 5))

    best_theta = center.copy()
    best_eval = float("inf")

    for g in range(int(gens)):
        solver.step()
        pop_batch = solver.population

        # decision vars
        vals_t = pop_batch.values if hasattr(pop_batch, "values") else pop_batch.access_values()
        vals = vals_t.detach().cpu().numpy()

        # evals (prefer pop.evals; fallback if needed)
        eval_source = "pop.evals"
        if hasattr(pop_batch, "evals") and (pop_batch.evals is not None):
            fits_t = pop_batch.evals
        elif hasattr(pop_batch, "access_evals"):
            fits_t = pop_batch.access_evals()
            eval_source = "access_evals()"
        else:
            fits_t = prob.evaluate(vals_t)
            eval_source = "forced_evaluate(vals_t)"

        fits = fits_t.detach().cpu().numpy().reshape(-1)
        i = int(np.argmin(fits))
        if fits[i] < best_eval:
            best_eval = float(fits[i])
            best_theta = vals[i].copy()

        # diagnostics
        console.log(
            f"[inner CMA gen {g:02d}] source={eval_source} pop={len(vals)} "
            f"min={np.min(fits):.4f} mean={np.mean(fits):.4f} max={np.max(fits):.4f}"
        )

    return best_theta


def decode_and_build(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray]):
    """Decode nde from a genotype and build robot"""
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core


def check_viability(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray], min_viable_movement: float):
    """Run a 6 sec random simulation and check if the robot has moved > threshold"""
    _, core = decode_and_build(nde, genotype)

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    simple_runner(model, data, duration=6, steps_per_loop=100)
    xpos_history = tracker.history.get("xpos", {})
    hist = xpos_history[0]

    pos_3 = hist[3]
    pos_final = hist[-1]
    pos_diff = pos_3 - pos_final
    viability = (abs(pos_diff[0]) > min_viable_movement or abs(pos_diff[1]) > min_viable_movement)

    return viability, hist


# ---------- Evaluation (train NA-CPG per body) ----------
def evaluate(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray], sim_time: float) -> tuple[float, "DiGraph", np.ndarray]:
    """Check viability -> if viable train NA-CPG via CMA-ES -> simulate with tracker -> fitness."""
    # Check viability
    viable, _ = check_viability(nde, genotype, MIN_VIABLE_MOVEMENT)
    if not viable:
        console.log("[bold red]Body was not viable, skipping CPG training[/bold red]")
        return -10, None, None  # very low fitness for non-viable

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    # Compile model for THIS body
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()

    # Inner-loop: train NA-CPG for this specific body
    theta = optimize_cpg_cma_for_body(model, seconds=sim_time, pop=CPG_TRAINING_POP, gens=CPG_TRAINING_GENS)

    # Re-run tracked experiment using trained NA-CPG
    def na_cpg_callback_factory(m: mj.MjModel) -> Any:
        nu = int(m.nu)
        phases = np.clip(theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX) if nu > 0 else np.zeros(0, dtype=np.float64)
        AMP  = float(np.clip(theta[nu] if nu > 0 else 0.8, NA_AMP_MIN, NA_AMP_MAX))
        FREQ = float(np.clip(theta[nu + 1] if nu > 0 else 1.5, NA_FREQ_MIN, NA_FREQ_MAX))
        omega = 2.0 * math.pi * FREQ

        cpg = BodyAgnosticNACPG.from_model(m, alpha=0.45, seed=SEED)
        with torch.inference_mode():
            if nu > 0:
                cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
                cpg.amplitudes[:] = AMP
                cpg.w[:] = omega
        return cpg.control_callback(m)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=na_cpg_callback_factory(model), tracker=tracker, alpha=CTRL_ALPHA)

    # Rebuild core/spec fresh for the tracked replay (avoids compile reuse issues)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    experiment(robot=core, controller=ctrl, duration=sim_time)

    # Use original fitness on the tracked history
    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return fit, robot_graph, theta


# ---------- Initialize viable population ----------
def initialize_viable_population(nde: NeuralDevelopmentalEncoding, pop_size: int) -> list[np.ndarray]:
    population = []
    while len(population) < pop_size:
        geno = random_genotype()
        viable, _ = check_viability(nde, geno, MIN_VIABLE_MOVEMENT)
        if viable:
            population.append(geno)
    return population


# ---------- Probabilistic parent selection ----------
def pick_parents(population: list[np.ndarray], fitnesses: np.ndarray, beta: float):
    """Pick one parent from population with probability exp(beta * fitness)."""
    shifted = fitnesses - fitnesses.min()
    probs = np.exp(beta * shifted)
    probs /= probs.sum()
    idx1 = RNG.choice(len(population), p=probs)
    # try up to 10 times to pick a different parent
    for _ in range(10):
        idx2 = RNG.choice(len(population), p=probs)
        if idx2 != idx1:
            break
    else:
        idx2 = idx1
        print(f"[yellow]Warning: Could not pick distinct parent after 10 attempts, using same parent twice[/yellow]")

    return population[idx1], population[idx2]


# ---------- EA main ----------
def main() -> None:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)  # Set NDE once, constant for entire run
    console.log(f"[bold cyan]Starting EA with NA-CPG (CMA-ES) for {N_GEN} generations, pop={POP_SIZE}[/bold cyan]")

    population = initialize_viable_population(nde, POP_SIZE)
    best_fit = -np.inf
    best_graph = None
    best_theta = None

    # Setup best fitness output csv file
    fitness_folder = DATA / "fitness_per_gen"
    fitness_folder.mkdir(exist_ok=True)
    fitness_file = fitness_folder / f"fitness_{TIMESTAMP}.csv"
    with open(fitness_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness"])

    # Main EA loop
    for gen in range(N_GEN):
        console.rule(f"[bold green]Generation {gen}[/bold green]")
        sim_time = get_sim_time_for_gen(gen)
        fitnesses = np.zeros(POP_SIZE)

        for i, geno in enumerate(population):
            fit, graph, theta = evaluate(nde, geno, sim_time)
            fitnesses[i] = fit
            console.log(f"Robot {i:02d} → fitness = {fit:.4f} (sim time {sim_time:.1f}s)")
            if fit > best_fit and graph is not None and theta is not None:
                best_fit, best_graph, best_theta = fit, graph, theta

        # Parent selection (elitism and probabilistic)
        sorted_idx = np.argsort(fitnesses)
        elite_idx = sorted_idx[-ELITISM_SIZE:]  # top-n elites kept
        elites = [population[i] for i in elite_idx]
        console.log(f"Best fitness={fitnesses[elite_idx[-1]]:.4f}")
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = pick_parents(population, fitnesses, beta=PICK_PARENTS_BETA)
            child = crossover_per_chromosome(p1, p2, CX_PROB)[0]  # only use one child
            child = gaussian_mutation(child, MUT_PROB, MUT_SIGMA)
            new_pop.append(child)
        population = new_pop

        # Save best fitness in csv
        with open(fitness_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fit])

        # Save video at 4 checkpoints
        if (gen + 1) % (N_GEN // 4) == 0 and best_graph is not None and best_theta is not None:
            console.log(f"[yellow]Checkpoint: recording video at generation {gen+1}[/yellow]")
            best_core = construct_mjspec_from_graph(best_graph)

            def na_cpg_video_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
                # Build NA-CPG once per callback install
                nu = int(m.nu)
                phases = np.clip(best_theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX) if nu > 0 else np.zeros(0, dtype=np.float64)
                AMP  = float(np.clip(best_theta[nu] if nu > 0 else 0.8, NA_AMP_MIN, NA_AMP_MAX))
                FREQ = float(np.clip(best_theta[nu + 1] if nu > 0 else 1.5, NA_FREQ_MIN, NA_FREQ_MAX))
                omega = 2.0 * math.pi * FREQ

                # cache on closure
                if not hasattr(na_cpg_video_controller, "_cb"):
                    cpg = BodyAgnosticNACPG.from_model(m, alpha=0.45, seed=SEED)
                    with torch.inference_mode():
                        if nu > 0:
                            cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
                            cpg.amplitudes[:] = AMP
                            cpg.w[:] = omega
                    na_cpg_video_controller._cb = cpg.control_callback(m)

                # Return target; Controller(alpha=0.6) will smooth
                cb = na_cpg_video_controller._cb
                return cb(m, d)

            tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            ctrl = Controller(controller_callback_function=na_cpg_video_controller, tracker=tracker, alpha=CTRL_ALPHA)

            video_folder = DATA / "videos"
            video_folder.mkdir(exist_ok=True)
            video_file = video_folder / f"video_gen{gen+1}_{TIMESTAMP}.mp4"
            experiment(robot=best_core, controller=ctrl, duration=sim_time, record=True)
            console.log(f"[green]Saved checkpoint video → {video_file}[/green]")

    # ---------- After final generation ----------
    console.rule("[bold magenta]Final best robot[/bold magenta]")
    console.log(f"Best fitness = {best_fit:.4f}")

    # Save graph JSON
    if best_graph is not None:
        graph_folder = DATA / "best_robot_graphs"
        graph_folder.mkdir(exist_ok=True)
        graph_file = f"best_robot_{TIMESTAMP}.json"
        save_graph_as_json(best_graph, graph_folder / graph_file)
        print(f"\nSaved best robot graph to {graph_folder / graph_file}")

    # Save video for best robot with trained NA-CPG
    if best_graph is not None and best_theta is not None:
        best_core = construct_mjspec_from_graph(best_graph)

        def na_cpg_video_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            nu = int(m.nu)
            phases = np.clip(best_theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX) if nu > 0 else np.zeros(0, dtype=np.float64)
            AMP  = float(np.clip(best_theta[nu] if nu > 0 else 0.8, NA_AMP_MIN, NA_AMP_MAX))
            FREQ = float(np.clip(best_theta[nu + 1] if nu > 0 else 1.5, NA_FREQ_MIN, NA_FREQ_MAX))
            omega = 2.0 * math.pi * FREQ

            if not hasattr(na_cpg_video_controller, "_cb"):
                cpg = BodyAgnosticNACPG.from_model(m, alpha=0.45, seed=SEED)
                with torch.inference_mode():
                    if nu > 0:
                        cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
                        cpg.amplitudes[:] = AMP
                        cpg.w[:] = omega
                na_cpg_video_controller._cb = cpg.control_callback(m)

            cb = na_cpg_video_controller._cb
            return cb(m, d)

        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        ctrl = Controller(controller_callback_function=na_cpg_video_controller, tracker=tracker, alpha=CTRL_ALPHA)
        console.log("[yellow]Recording video of best robot...[/yellow]")
        video_folder = DATA / "videos"
        video_folder.mkdir(exist_ok=True)
        experiment(robot=best_core, controller=ctrl, duration=get_sim_time_for_gen(N_GEN - 1), record=True)
        console.log(f"[green]All done! Video and graph saved.[/green], at {video_folder}")
    else:
        console.log("[red]No viable best_graph/best_theta to render.[/red]")


if __name__ == "__main__":
    main()
