"""Assignment 3 – Minimal GA + per-body CMA-ES NA-CPG + export of best robot video and JSON.

This file keeps the teammate’s outer EA (population, crossover, mutation, elitism, selection,
staged sim times, CSV, videos, JSON) EXACTLY the same, and replaces the inner-loop controller
with the Body-Agnostic NA-CPG (one oscillator per actuator) trained by CMA-ES per body.

Inner-loop CMA-ES optimizes theta = [phase_0..phase_{nu-1}, AMP, FREQ].

NA-CPG parameters & behavior (updated):
- Controller smoothing inside Controller: alpha = 0.6
- Internal oscillator alpha ≈ 0.45, COUP = 0.08
- Defaults: w[:] = 2π*1.5 Hz, phase[:] = 0.0
- NEW AMP semantics (recommended): AMP ∈ [0,1] = fraction of each joint’s half-range
  Mapping: y_unit ∈ [-1,1] → target = center + half * (AMP * y_unit), then clip to ctrlrange
  This allows using the **entire actuator range** cleanly when AMP → 1.
- Bounds on theta: phase ∈ [-π, π], AMP ∈ [0.0, 1.0], FREQ ∈ [0.6, 3.0]
"""

# ---------- Imports ----------
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Dict, Tuple, Optional
import math
import json
import numpy as np
import numpy.typing as npt
import mujoco as mj
import csv
from datetime import datetime

# [PARALLEL]
from concurrent.futures import ProcessPoolExecutor
import os

import torch
from torch import nn
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.core import SolutionBatch
from evotorch.logging import StdOutLogger  # harmless if unused

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

try:
    from mujoco import viewer as mjviewer  # optional viewer for quick testing
except Exception:
    mjviewer = None

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
POP_SIZE = 14
N_GEN = 15
CX_PROB = 0.6
MUT_PROB = 0.5
MUT_SIGMA = 0.25
ELITISM_SIZE = 3
PICK_PARENTS_BETA = 5
if N_GEN > 30:
    SIM_TIME_STAGES = [10.0, 20.0, 40.0, 60.0]
else:
    SIM_TIME_STAGES = [10.0, 10.0, 15.0, 20.0]

# Inner EA loop parameters (per-body CMA-ES)
MIN_VIABLE_MOVEMENT = 0.015
CPG_TRAINING_POP = 16
CPG_TRAINING_GENS = 10

# Teammate globals (kept; NA-CPG enforces its own):
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
AMP_MIN, AMP_MAX     = 0.0, 1.0
FREQ_MIN, FREQ_MAX   = 0.4, 2.0
SMOOTH_ALPHA         = 0.5  # kept, but NA-CPG replay uses Controller(alpha=0.6)

# NA-CPG controller smoothing (as per user spec)
CTRL_ALPHA = 0.6

# NA-CPG theta bounds (AMP is fraction of half-range)
NA_PHASE_MIN, NA_PHASE_MAX = -math.pi, math.pi
NA_AMP_MIN,   NA_AMP_MAX   = 0.0, 1.0
NA_FREQ_MIN,  NA_FREQ_MAX  = 0.6, 3.0

# Optional viewer
LAUNCH_VIEWER = True  # set False for headless runs

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
def experiment(robot: Any, controller: Controller, duration: float, record: bool = False,
               warmup_seconds: float = 0.75) -> None:
    """Run a sim with an optional warm-up so CPG reaches its limit cycle before we track/record."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Attach tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Install controller callback (Controller handles smoothing)
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    # --- Warm-up (no saving, no recording) ---
    if warmup_seconds and warmup_seconds > 0:
        steps_per_sec = int(round(1.0 / model.opt.timestep))
        n_warm = max(1, int(round(warmup_seconds * steps_per_sec)))
        for _ in range(n_warm):
            mj.mj_step(model, data)
        # Clear tracker history after warm-up
        if controller.tracker is not None:
            controller.tracker.reset()

    # --- Main run ---
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

    IMPORTANT: step() returns *unit* oscillator output in [-1,1] (no AMP, no π/2 scaling).
               control_callback() does the only scaling: center + half * (AMP * y_unit).
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
        # amplitudes now used as a FRACTION of half-range (0..1), but only in control mapping (not in step)
        self.amplitudes = nn.Parameter(torch.full((self.n,), 1.0), requires_grad=False)
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
        """Advance oscillators one MuJoCo step and return *unit* outputs y_unit ∈ [-1, 1] per joint."""
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

        # Use y as oscillator output; just bound to [-1,1]. No AMP here.
        y_unit = torch.clamp(self.xy[:, 1], -1.0, 1.0).detach().cpu().numpy()
        return y_unit

    def control_callback(self, model: mj.MjModel) -> Any:
        """
        Returns a target-only callback. TARGET MAPPING NOW USES JOINT RANGE, not actuator ctrlrange.

        For each actuator:
          - find its driven hinge joint via actuator_trnid
          - use that joint's jnt_range as [lo_joint, hi_joint]
          - if range is degenerate/unlimited, default to [-π/2, +π/2]
        Final target = center_joint + half_joint * (AMP * y_unit), then clip to actuator_ctrlrange.
        """
        # --- Resolve actuator->joint mapping and ranges (cached once) ---
        if not hasattr(self, "_map_built"):
            nu = int(model.nu)
            self._lo_joint = np.zeros(nu, dtype=np.float64)
            self._hi_joint = np.zeros(nu, dtype=np.float64)
            self._lo_ctrl  = model.actuator_ctrlrange[:, 0].astype(np.float64)
            self._hi_ctrl  = model.actuator_ctrlrange[:, 1].astype(np.float64)

            # actuator_trnid: shape (nu, 2), first entry is joint id when actuator targets a joint
            trn = model.actuator_trnid
            for a in range(nu):
                j = int(trn[a, 0])
                # Guard: invalid target or non-hinge → default to ±π/2
                use_default = True
                if 0 <= j < model.njnt:
                    if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
                        lo, hi = model.jnt_range[j]
                        # If unlimited or tiny span, fall back
                        if np.isfinite(lo) and np.isfinite(hi) and (abs(hi - lo) > 1e-6):
                            self._lo_joint[a] = float(lo)
                            self._hi_joint[a] = float(hi)
                            use_default = False
                if use_default:
                    self._lo_joint[a] = -math.pi / 2.0
                    self._hi_joint[a] = +math.pi / 2.0

            # no range dump; keep quiet
            self._map_built = True

        lo_j, hi_j = self._lo_joint, self._hi_joint
        lo_c, hi_c = self._lo_ctrl,  self._hi_ctrl
        center_j = 0.5 * (hi_j + lo_j)
        half_j   = 0.5 * (hi_j - lo_j)

        def _cb(_m: mj.MjModel, _d: mj.MjData) -> npt.NDArray[np.float64]:
            # oscillator output in [-1,1]
            y_unit = self.step()
            # AMP is fraction of half-joint-range
            amp = float(self.amplitudes[0].detach().cpu().item()) if self.amplitudes.numel() > 0 else 1.0
            amp = float(np.clip(amp, NA_AMP_MIN, NA_AMP_MAX))
            # intent uses joint range
            target = center_j + half_j * (amp * y_unit)
            # safety clip to actuator ctrlrange (should be ≥ joint span if actuator allows it)
            return np.clip(target, lo_c, hi_c)

        return _cb


# ---------- Controller factory for replay/video ----------
def make_na_cpg_controller_for_video(best_theta: np.ndarray, seed: int = SEED):
    """
    Builds NA-CPG on first call and returns target controls (Controller handles smoothing).
    Minimal one-time diagnostics: nu, dt, steps_per_sec, AMP, FREQ, omega.
    """
    state: dict[str, Any] = {"cb": None}

    def _video_cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        if (state["cb"] is None) or (d.time == 0.0):
            nu = int(m.nu)
            if nu == 0:
                return np.zeros(0, dtype=np.float64)

            phases = np.clip(best_theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX)
            AMP    = float(np.clip(best_theta[nu],     NA_AMP_MIN,  NA_AMP_MAX))   # fraction of half-range
            FREQ   = float(np.clip(best_theta[nu + 1], NA_FREQ_MIN, NA_FREQ_MAX))
            omega  = 2.0 * math.pi * FREQ

            cpg = BodyAgnosticNACPG.from_model(m, alpha=0.45, seed=seed)
            with torch.inference_mode():
                cpg.phase[:]      = torch.from_numpy(phases.astype(np.float32))
                cpg.amplitudes[:] = AMP
                cpg.w[:]          = omega

            # ---- minimal sanity print (no ctrlrange) ----
            dt = float(m.opt.timestep)
            steps_per_sec = int(round(1.0 / dt))
            console.log(
                "nu="
                + str(nu)
                + f"  dt={dt:.5f}s  steps_per_sec≈{steps_per_sec}  "
                  f"AMP={AMP:.3f}  FREQ={FREQ:.3f}  omega={omega:.3f}"
            )

            state["cb"] = cpg.control_callback(m)

        return state["cb"](m, d)

    return _video_cb


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

        # NA-CPG bounds (AMP fractional)
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
        AMP  = float(np.clip(theta[nu],     NA_AMP_MIN,  NA_AMP_MAX))   # fraction
        FREQ = float(np.clip(theta[nu + 1], NA_FREQ_MIN, NA_FREQ_MAX))
        omega = 2.0 * math.pi * FREQ

        # Build NA-CPG
        cpg = BodyAgnosticNACPG.from_model(self.model, alpha=0.45, seed=SEED)
        with torch.inference_mode():
            cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
            cpg.amplitudes[:] = AMP
            cpg.w[:] = omega

        # RAW MuJoCo control callback (no Controller, no tracker, no extra smoothing)
        raw_cb = cpg.control_callback(self.model)

        def mjcb(_m: mj.MjModel, d: mj.MjData):
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

    # Better center: small phase jitter, AMP near full half-range
    phase_center = RNG.uniform(-0.2, 0.2, size=nu) if nu > 0 else np.zeros(0, dtype=np.float64)
    center = np.concatenate([phase_center, [0.95], [1.5]]).astype(np.float64)

    solver = CMAES(
        prob,
        popsize=max(10, int(pop)),
        stdev_init=0.35,
        center_init=torch.from_numpy(center),
    )
    # No StdOutLogger: keep output minimal

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

        # minimal inner-loop progress line
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


# ---------- [PARALLEL] graph-based viability for workers ----------
def check_viability_from_graph(robot_graph, min_viable_movement: float):
    """Run a 6s random simulation on a decoded graph and check if movement > threshold."""
    core = construct_mjspec_from_graph(robot_graph)

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

    # Compile model for THIS body (training happens on this model)
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()

    # Inner-loop: train NA-CPG for this specific body
    theta = optimize_cpg_cma_for_body(model, seconds=sim_time, pop=CPG_TRAINING_POP, gens=CPG_TRAINING_GENS)

    # Re-run tracked experiment using trained NA-CPG
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # single-smoothing: Controller handles low-pass at explicit control rate (~100 Hz)
    steps_per_sec = int(round(1.0 / model.opt.timestep))
    steps_per_ctrl = max(1, steps_per_sec // 100)
    console.log(f"[sanity] Controller alpha={CTRL_ALPHA}  steps_per_ctrl={steps_per_ctrl} (~{steps_per_sec/steps_per_ctrl:.1f} Hz)")

    ctrl = Controller(
        controller_callback_function=make_na_cpg_controller_for_video(theta, seed=SEED),
        tracker=tracker,
        alpha=CTRL_ALPHA,
        time_steps_per_ctrl_step=steps_per_ctrl,
    )

    # Rebuild core/spec fresh for the tracked replay (avoids compile reuse issues)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    experiment(robot=core, controller=ctrl, duration=sim_time)

    # Use original fitness on the tracked history
    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return fit, robot_graph, theta


# ---------- [PARALLEL] worker for graph -> fitness, theta ----------
def _evaluate_worker_from_graph(args):
    """
    args: (robot_graph, sim_time, seed_offset)
    Returns: (fitness: float, robot_graph, theta: np.ndarray|None)
    """
    robot_graph, sim_time, seed_offset = args

    # Keep workers lean on CPU threading
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Viability check on decoded graph
    viable, _ = check_viability_from_graph(robot_graph, MIN_VIABLE_MOVEMENT)
    if not viable:
        return -10.0, robot_graph, None

    # Build model in worker
    mj.set_mjcb_control(None)
    core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()

    # Train NA-CPG
    theta = optimize_cpg_cma_for_body(
        model, seconds=sim_time, pop=CPG_TRAINING_POP, gens=CPG_TRAINING_GENS
    )

    # Replay with Controller(alpha=0.6) + Tracker("core") at explicit control rate
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    steps_per_sec = int(round(1.0 / model.opt.timestep))
    steps_per_ctrl = max(1, steps_per_sec // 100)
    console.log(f"[sanity] (worker) Controller alpha={CTRL_ALPHA}  steps_per_ctrl={steps_per_ctrl} (~{steps_per_sec/steps_per_ctrl:.1f} Hz)")

    ctrl = Controller(
        controller_callback_function=make_na_cpg_controller_for_video(theta, seed=SEED + int(seed_offset)),
        tracker=tracker,
        alpha=CTRL_ALPHA,
        time_steps_per_ctrl_step=steps_per_ctrl,
    )
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    try:
        simple_runner(model, data, duration=sim_time)
    finally:
        mj.set_mjcb_control(None)

    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return float(fit), robot_graph, theta


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
        console.log(f"[outer EA] gen={gen}/{N_GEN-1} sim_time={sim_time:.1f}s pop={len(population)}")

        # [PARALLEL] Pre-decode ALL genotypes to graphs in the MAIN process
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        decoded_graphs = []
        for geno in population:
            p_type, p_conn, p_rot = nde.forward(geno)
            G = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
            decoded_graphs.append(G)

        # [PARALLEL] Evaluate across bodies in parallel processes
        MAX_WORKERS = max(1, os.cpu_count() - 1)
        results = [None] * len(population)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = []
            for i, G in enumerate(decoded_graphs):
                seed_offset = gen * POP_SIZE + i
                futures.append(ex.submit(_evaluate_worker_from_graph, (G, sim_time, seed_offset)))
            for i, fut in enumerate(futures):
                results[i] = fut.result()

        # Unpack results, record best, print as before
        fitnesses = np.zeros(POP_SIZE)
        for i, (fit, graph, theta) in enumerate(results):
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
        if (N_GEN >= 4) and ((gen + 1) % (N_GEN // 4) == 0) and best_graph is not None and best_theta is not None:
            console.log(f"[yellow]Checkpoint: recording video at generation {gen+1}[/yellow]")
            best_core = construct_mjspec_from_graph(best_graph)

            # compile here to compute correct control rate
            mj.set_mjcb_control(None)
            world = OlympicArena()
            world.spawn(best_core.spec, position=SPAWN_POS)
            model = world.spec.compile()
            data = mj.MjData(model)

            tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            tracker.setup(world.spec, data)

            steps_per_sec = int(round(1.0 / float(model.opt.timestep)))
            steps_per_ctrl = max(1, steps_per_sec // 100)

            ctrl = Controller(
                controller_callback_function=make_na_cpg_controller_for_video(best_theta, seed=SEED),
                tracker=tracker,
                alpha=CTRL_ALPHA,
                time_steps_per_ctrl_step=steps_per_ctrl,
            )

            video_folder = DATA / "videos"
            video_folder.mkdir(exist_ok=True)
            recorder = VideoRecorder(output_folder=str(video_folder))
            console.log("[yellow]Recording checkpoint video…[/yellow]")
            video_renderer(model, data, duration=sim_time, video_recorder=recorder)
            console.log(f"[green]Saved checkpoint video to {video_folder}[/green]")

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

    # Save video for best robot with trained NA-CPG + viewer (90 seconds)
    if best_graph is not None and best_theta is not None:
        best_core_vid = construct_mjspec_from_graph(best_graph)

        def na_cpg_video_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            nu = int(m.nu)
            phases = np.clip(best_theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX) if nu > 0 else np.zeros(0, dtype=np.float64)
            AMP  = float(np.clip(best_theta[nu] if nu > 0 else 1.0, NA_AMP_MIN, NA_AMP_MAX))  # fraction
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
            return na_cpg_video_controller._cb(m, d)

        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        ctrl = Controller(
            controller_callback_function=na_cpg_video_controller,
            tracker=tracker,
            alpha=CTRL_ALPHA,
            time_steps_per_ctrl_step=1,  # experiment() recomputes from compiled dt
        )

        console.log("[yellow]Recording video of best robot (90s)…[/yellow]")
        experiment(robot=best_core_vid, controller=ctrl, duration=90.0, record=True)
        console.log("[green]All done! Video and graph saved.[/green]")

        # -------- Interactive viewer (fresh spec again) --------
        try:
            console.log("[magenta]Launching interactive viewer with best θ… close the window to finish.[/magenta]")
            mj.set_mjcb_control(None)
            world = OlympicArena()
            best_core_view = construct_mjspec_from_graph(best_graph)  # fresh spec for viewer
            world.spawn(best_core_view.spec, position=SPAWN_POS)
            model = world.spec.compile()
            data = mj.MjData(model)

            tracker_v = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            tracker_v.setup(world.spec, data)

            def _viewer_cb_factory(m: mj.MjModel) -> Any:
                nu = int(m.nu)
                phases = np.clip(best_theta[:nu], NA_PHASE_MIN, NA_PHASE_MAX) if nu > 0 else np.zeros(0, dtype=np.float64)
                AMP  = float(np.clip(best_theta[nu] if nu > 0 else 0.8, NA_AMP_MIN, NA_AMP_MAX))
                FREQ = float(np.clip(best_theta[nu + 1] if nu > 0 else 1.5, NA_FREQ_MIN, NA_FREQ_MAX))
                omega = 2.0 * math.pi * FREQ
                cpg = BodyAgnosticNACPG.from_model(m, alpha=0.45, seed=SEED)
                with torch.inference_mode():
                    if nu > 0:
                        cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
                        cpg.amplitudes[:] = AMP
                        cpg.w[:] = omega
                return cpg.control_callback(m)

            ctrl_v = Controller(controller_callback_function=_viewer_cb_factory(model),
                                tracker=tracker_v, alpha=CTRL_ALPHA,
                                time_steps_per_ctrl_step=max(1, int(round(1.0 / float(model.opt.timestep))) // 100))
            mj.set_mjcb_control(lambda m, d: ctrl_v.set_control(m, d))

            try:
                from mujoco import viewer as mjviewer
                mjviewer.launch(model, data)  # blocks until window closed
            finally:
                mj.set_mjcb_control(None)
        except Exception as e:
            console.log(f"[red]Viewer launch failed: {e}[/red]")
    else:
        console.log("[red]No viable best_graph/best_theta to render.[/red]")


if __name__ == "__main__":
    main()
