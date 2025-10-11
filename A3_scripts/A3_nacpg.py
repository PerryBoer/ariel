# A3_nacpg.py
# CMA-ES training of a Body-Agnostic NA-CPG on GECKO (toggle) or one random robot in OlympicArena.
# - Evolves: phases_i (per actuator), A_global, f_global
# - Fitness: negative 3D distance to TARGET_POSITION
# - Prints per-gen diagnostics; replays EXACT best rollout; saves CSV + video; launches viewer.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import math
import numpy as np
import numpy.typing as npt
import mujoco as mj
from mujoco import viewer
import torch
from torch import nn

# Ariel
from ariel import console
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# EvoTorch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.core import SolutionBatch
from evotorch.logging import StdOutLogger

# ---------- Paths & constants ----------
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "videos").mkdir(parents=True, exist_ok=True)

SPAWN_POS = [-0.8, 0.0, 0.1]
NUM_OF_MODULES = 30
GENOTYPE_SIZE = 64

SIM_SECONDS = 10.0
TARGET_POSITION = np.array([5.0, 0.0, 0.5], dtype=float)

# CMA-ES
POPSIZE = 32 # 32
GENERATIONS = 10 # 5
SIGMA_INIT = 0.6

# Controller hyperparams
CTRL_HZ = 100.0
SAVE_HZ = 5.0
CTRL_ALPHA = 0.5

# Early termination inside evolution (disabled for final replay)
EARLY_STOP = True
FALL_Z_THRESH = 0.08
FALL_PERSIST_STEPS = 0.25  # seconds


# ---------------- helpers ----------------
def create_fully_connected_adjacency(n: int) -> Dict[int, List[int]]:
    return {i: [j for j in range(n) if j != i] for i in range(n)}


def random_genotype() -> List[np.ndarray]:
    return [
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
    ]


def build_random_body(nde: NeuralDevelopmentalEncoding):
    geno = random_genotype()
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(geno)
    G = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(G)
    return core, G


# ---------------- NA-CPG ----------------
class BodyAgnosticNACPG(nn.Module):
    """
    One oscillator per actuator. Oscillator outputs are mapped to each joint's center ± half-span.
    """
    xy: torch.Tensor
    xy_dot_old: torch.Tensor
    angles: torch.Tensor

    def __init__(
        self,
        adjacency_dict: Dict[int, List[int]],
        alpha: float = 0.45,
        dt: float = 0.01,
        hard_bounds: Optional[Tuple[float, float]] = None,
        *,
        angle_tracking: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        self.alpha = alpha
        self.dt = dt
        self.hard_bounds = hard_bounds
        self.angle_tracking = angle_tracking
        self.clamping_error = 0.0

        # Evolvable params (set per rollout)
        scale = 2 * torch.pi
        self.phase = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.amplitudes = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.w = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)

        self.ha = nn.Parameter(torch.empty(self.n), requires_grad=False)
        self.b  = nn.Parameter(torch.empty(self.n),  requires_grad=False)
        with torch.no_grad():
            self.ha.fill_(0.25)
            self.b.fill_(0.0)

        # Buffers
        self.register_buffer("xy", torch.randn(self.n, 2))
        self.register_buffer("xy_dot_old", torch.zeros(self.n, 2))
        self.register_buffer("angles", torch.zeros(self.n))
        self.angle_history: List[List[float]] = []

        # Actuator ranges bound later
        self._ctrl_lo_np: Optional[np.ndarray] = None
        self._ctrl_hi_np: Optional[np.ndarray] = None

    @staticmethod
    def term_a(alpha: float, r2i: float) -> float:
        return alpha * (1 - r2i**2)

    @staticmethod
    def term_b(zeta_i: float, w_i: float) -> float:
        E = 1e-9
        return (1 / (zeta_i + E)) * w_i

    @staticmethod
    def zeta(ha_i: float, x_dot_old: float) -> float:
        E = 1e-9
        return 1 - ha_i * ((x_dot_old + E) / (torch.abs(x_dot_old) + E))

    def reset(self) -> None:
        with torch.inference_mode():
            self.xy.normal_(mean=0.0, std=0.05)
            self.xy_dot_old.zero_()
            self.angles.zero_()
            self.angle_history.clear()
            self.clamping_error = 0.0

    def forward(self, time: float | None = None) -> torch.Tensor:
        if time is not None and abs(float(time)) < 1e-12:
            self.reset()

        with torch.inference_mode():
            n = self.n
            r_matrix = torch.zeros(n, n, 2, 2)
            I2 = torch.eye(2)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        r_matrix[i, j] = I2
                    else:
                        d = self.phase[i] - self.phase[j]
                        c, s = torch.cos(d), torch.sin(d)
                        r_matrix[i, j] = torch.tensor([[c, -s], [s, c]])

            k_matrix = torch.zeros(n, 2, 2)
            for i in range(n):
                x_dot_old, _ = self.xy_dot_old[i]
                xi, yi = self.xy[i]
                r2i = xi * xi + yi * yi
                a = self.term_a(self.alpha, r2i)
                z = self.zeta(self.ha[i], x_dot_old)
                b = self.term_b(z, self.w[i])
                k_matrix[i] = torch.tensor([[a, -b], [b, a]])

            angles = torch.zeros(n)
            COUP = 0.08
            for i, (xi, yi) in enumerate(self.xy):
                local = torch.mv(k_matrix[i], self.xy[i])

                coup = torch.zeros(2)
                nbrs = self.adjacency_dict[i]
                if len(nbrs) > 0:
                    for j in nbrs:
                        coup += COUP * torch.mv(r_matrix[i, j], self.xy[j])
                    coup /= float(len(nbrs))

                xi_dot, yi_dot = local + coup

                xi_dot_old, yi_dot_old = self.xy_dot_old[i]
                diff = 10.0
                xi_dot = torch.clamp(xi_dot, xi_dot_old - diff, xi_dot_old + diff)
                yi_dot = torch.clamp(yi_dot, yi_dot_old - diff, yi_dot_old + diff)

                xi_new = xi + self.dt * xi_dot
                yi_new = yi + self.dt * yi_dot

                self.xy_dot_old[i] = torch.tensor([xi_dot, yi_dot])
                self.xy[i]         = torch.tensor([xi_new, yi_new])

                angles[i] = self.amplitudes[i] * yi_new + self.b[i]

            if self.hard_bounds is not None:
                pre = angles.clone()
                angles = torch.clamp(angles, self.hard_bounds[0], self.hard_bounds[1])
                self.clamping_error = float((pre - angles).abs().sum().item())

            if self.angle_tracking:
                self.angle_history.append(angles.clone().tolist())

            self.angles = angles
            return self.angles.clone()

    @classmethod
    def from_model(
        cls,
        model: mj.MjModel,
        *,
        alpha: float = 0.5,
        seed: Optional[int] = None,
        angle_tracking: bool = False,
    ) -> "BodyAgnosticNACPG":
        nu = int(model.nu)
        if nu <= 0:
            raise ValueError("Model has zero actuators.")
        adj = create_fully_connected_adjacency(nu)
        inst = cls(
            adjacency_dict=adj,
            alpha=alpha,
            dt=float(model.opt.timestep),
            hard_bounds=None,
            angle_tracking=angle_tracking,
            seed=seed,
        )
        with torch.no_grad():
            inst.phase[:]      = 0.0
            inst.amplitudes[:] = 0.8
            inst.w[:]          = 2.0 * math.pi * 2    # ~1.5 Hz
            inst.ha[:]         = 0.25
            inst.b[:]          = 0.0
        inst._bind_model_ranges(model)
        return inst

    def _bind_model_ranges(self, model: mj.MjModel) -> None:
        self._ctrl_lo_np = model.actuator_ctrlrange[:, 0].astype(np.float64)
        self._ctrl_hi_np = model.actuator_ctrlrange[:, 1].astype(np.float64)

    def mj_callback(self, model: mj.MjModel) -> Any:
        if self._ctrl_lo_np is None:
            self._bind_model_ranges(model)
        lo, hi = self._ctrl_lo_np, self._ctrl_hi_np
        center    = 0.5 * (hi + lo)
        half_span = 0.5 * (hi - lo)

        def _cb(_m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            if abs(d.time) < 1e-12:
                self.reset()
            a = self.forward(float(d.time)).detach().cpu().numpy()
            norm = np.clip(a / (np.pi / 2.0), -1.0, 1.0)
            target = center + half_span * norm
            return np.clip(target, lo, hi)

        return _cb


# ---------------- Fitness ----------------
def fitness_function(history: List[Tuple[float, float, float]]) -> float:
    if not history:
        return -1e6
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = float(np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2))
    if not np.isfinite(cartesian_distance):
        return -1e6
    return -cartesian_distance


# ---------------- Problem ----------------
class OneBodyCPGProblem(Problem):
    def __init__(self, use_gecko: bool = False):
        mj.set_mjcb_control(None)
        self.world = OlympicArena()

        if use_gecko:
            console.rule("[bold green]Using prebuilt GECKO in OlympicArena[/bold green]")
            core_spec = gecko().spec
            self.world.spawn(core_spec, position=SPAWN_POS)
            self.model = self.world.spec.compile()
            self.data = mj.MjData(self.model)
            self.core = None
        else:
            console.rule("[bold green]Build one random robot in OlympicArena[/bold green]")
            nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
            self.core, G = build_random_body(nde)
            save_graph_as_json(G, DATA / "robot_graph.json")
            console.log(f"Saved robot graph → {DATA / 'robot_graph.json'}")
            self.world.spawn(self.core.spec, position=SPAWN_POS)
            self.model = self.world.spec.compile()
            self.data = mj.MjData(self.model)

        # Timing & sizes
        self.steps_per_sec = int(round(1.0 / float(self.model.opt.timestep)))
        self.nu = int(self.model.nu)
        head = min(4, self.nu)
        print(f"[Model] dt={self.model.opt.timestep:.5f}s  steps/sec≈{self.steps_per_sec}  nu={self.nu}")
        print("[Actuators] ctrlrange[:4]=\n", np.round(self.model.actuator_ctrlrange[:head, :], 3))

        # Tracker (track core geoms)
        self.tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        self.tracker.setup(self.world.spec, self.data)
        if isinstance(self.tracker.history.get("xpos", {}), dict):
            print(f"[Tracker] tracking slots (xpos): {len(self.tracker.history['xpos'])}")

        # Precompute core geom ids for early-stop
        self.core_geom_ids: List[int] = []
        for gid in range(self.model.ngeom):
            try:
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, gid)
            except Exception:
                name = None
            if name and ("core" in name):
                self.core_geom_ids.append(gid)
        if not self.core_geom_ids:
            self.core_geom_ids = [0]

        # Controller + CPG
        self.cpg = BodyAgnosticNACPG.from_model(self.model, alpha=0.8, seed=SEED, angle_tracking=False)
        steps_per_ctrl = max(1, int(round(self.steps_per_sec / max(1e-6, CTRL_HZ))))
        steps_per_save = max(1, int(round(self.steps_per_sec / max(1e-6, SAVE_HZ))))
        print(f"[Controller] steps_per_ctrl={steps_per_ctrl} (~{CTRL_HZ} Hz)  steps_per_save={steps_per_save} (~{SAVE_HZ} Hz)  alpha={CTRL_ALPHA}")

        self.controller = Controller(
            controller_callback_function=self.cpg.mj_callback(self.model),
            tracker=self.tracker,
            time_steps_per_ctrl_step=steps_per_ctrl,
            time_steps_per_save=steps_per_save,
            alpha=CTRL_ALPHA,
        )
        mj.set_mjcb_control(lambda m, d: self.controller.set_control(m, d))

        # Genome: [phi_0..phi_{nu-1}, A_global, f_global]
        nu = self.nu
        L = nu + 2
        lo = np.concatenate([np.full(nu, -math.pi), [0.4], [1.0]]).astype(np.float64)
        hi = np.concatenate([np.full(nu,  math.pi), [0.8], [2.0]]).astype(np.float64)

        super().__init__(
            solution_length=L,
            initial_bounds=(torch.from_numpy(lo), torch.from_numpy(hi)),
            objective_sense="max",
            dtype=torch.float64,
            device="cpu",
        )

    # Extract trajectory from Ariel tracker
    def _extract_traj(self) -> List[Tuple[float, float, float]]:
        xhist = self.tracker.history.get("xpos", {})
        if isinstance(xhist, dict) and len(xhist) > 0:
            traj_list = xhist.get(0, [])
        else:
            traj_list = xhist[0] if isinstance(xhist, list) and xhist else []
        traj = [(float(p[0]), float(p[1]), float(p[2])) for p in traj_list] if traj_list else []
        return traj

    def _early_stop_fallen(self, t_steps: int) -> bool:
        if not EARLY_STOP:
            return False
        z_vals = [float(self.data.geom_xpos[gid][2]) for gid in self.core_geom_ids]
        z_min = min(z_vals) if z_vals else float("inf")
        if z_min < FALL_Z_THRESH:
            persist = int(round(FALL_PERSIST_STEPS * self.steps_per_sec))
            return t_steps >= persist
        return False

    def _rollout_fitness(self, theta: np.ndarray) -> Tuple[float, List[Tuple[float, float, float]], dict]:
        nu = self.nu
        phases = theta[:nu]
        A = float(theta[nu])
        f = float(theta[nu + 1])
        omega = 2.0 * math.pi * f

        with torch.inference_mode():
            self.cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
            self.cpg.amplitudes[:] = A
            self.cpg.w[:] = omega

        # Reset
        self.tracker.reset()
        mj.mj_resetData(self.model, self.data)
        if self.data.ctrl is not None:
            self.data.ctrl[:] = 0.0

        # Sim loop
        horizon = int(round(SIM_SECONDS * self.steps_per_sec))
        clamp_hits = 0
        total_outputs = horizon * nu

        t = 0
        while t < horizon:
            mj.mj_step(self.model, self.data)
            if self.data.ctrl is not None and self.cpg._ctrl_lo_np is not None:
                lo, hi = self.cpg._ctrl_lo_np, self.cpg._ctrl_hi_np
                u = np.asarray(self.data.ctrl, dtype=np.float64)
                clamp_hits += int(np.sum((u <= lo + 1e-9) | (u >= hi - 1e-9)))
            if self._early_stop_fallen(t):
                break
            t += 1

        traj = self._extract_traj()
        fit = fitness_function(traj)

        diags = {
            "A": A,
            "f": f,
            "omega": omega,
            "phase_std": float(np.std(phases)),
            "clamp_ratio": float(clamp_hits) / float(max(1, total_outputs)),
            "cpg_clamp_error": self.cpg.clamping_error,
            "distance": -fit if np.isfinite(fit) else float("inf"),
            "rolled_steps": t,
            "rolled_time_s": t / float(self.steps_per_sec),
        }
        return fit, traj, diags

    def evaluate(self, X):
        if isinstance(X, SolutionBatch):
            vals = X.access_values()
            fits = []
            for row in vals:
                theta = row.detach().cpu().numpy()
                fit, _, _ = self._rollout_fitness(theta)
                fits.append(fit)
            fits_t = torch.as_tensor(fits, dtype=vals.dtype, device=vals.device)
            X.set_evals(fits_t)
            return fits_t
        elif isinstance(X, torch.Tensor):
            return torch.as_tensor(
                [self._rollout_fitness(row.detach().cpu().numpy())[0] for row in X],
                dtype=X.dtype, device=X.device,
            )
        else:
            raise TypeError(f"Unsupported input to evaluate(): {type(X)}")


# ---------------- Train, record, viewer ----------------
def main():
    GECKO = True   # <<< toggle here (True: gecko, False: random)
    title = "GECKO" if GECKO else "Random Robot"
    console.rule(f"[bold green]CMA-ES: Body-Agnostic NA-CPG on {title}[/bold green]")

    problem = OneBodyCPGProblem(use_gecko=GECKO)
    nu = problem.nu

    solver = CMAES(problem, popsize=POPSIZE, stdev_init=SIGMA_INIT)
    _ = StdOutLogger(solver, interval=1)

    best_theta: Optional[np.ndarray] = None
    best_fit: float = -float("inf")
    best_diags: Optional[dict] = None

    # --- Training loop (ALWAYS captures the best θ from population) ---
    for gen in range(GENERATIONS):
        solver.step()

        pop = solver.population

        # ---- values (decision variables)
        if hasattr(pop, "values"):          # preferred if present
            vals_t = pop.values             # (popsize, L)
        else:
            vals_t = pop.access_values()    # legacy accessor
        vals = vals_t.detach().cpu().numpy()

        # ---- evals (fitnesses) across evotorch variants
        evals = None
        eval_source = "none"

        # 1) modern
        if hasattr(pop, "evals") and pop.evals is not None:
            evals_t = pop.evals
            evals = evals_t.detach().cpu().numpy().reshape(-1)
            eval_source = "pop.evals"

        # 2) legacy accessors
        if (evals is None) or (evals.size == 0) or (not np.isfinite(evals).any()):
            if hasattr(pop, "access_evals"):
                evals_t = pop.access_evals()
                if evals_t is not None:
                    evals = evals_t.detach().cpu().numpy().reshape(-1)
                    eval_source = "access_evals()"
            elif hasattr(pop, "get_evals"):
                evals_t = pop.get_evals()
                if evals_t is not None:
                    evals = evals_t.detach().cpu().numpy().reshape(-1)
                    eval_source = "get_evals()"

        # 3) status.pop_best (keep training even if full vector missing)
        status = solver.status or {}
        pop_best = status.get("pop_best", None)
        theta_pb = None
        fit_pb = -float("inf")
        if ((evals is None) or (evals.size == 0) or (not np.isfinite(evals).any())) and (pop_best is not None):
            theta_pb = pop_best.values.detach().cpu().numpy()
            # some builds use .eval, some .evaluation, some .fitness
            fit_pb = float(getattr(pop_best, "eval", getattr(pop_best, "evaluation",
                        getattr(pop_best, "fitness", np.nan))))
            eval_source = "status.pop_best"

        # 4) last resort: explicitly evaluate full population
        if (evals is None) or (evals.size == 0) or (not np.isfinite(evals).any()):
            eval_source = "forced_evaluate(vals_t)"
            evals = problem.evaluate(vals_t).detach().cpu().numpy().reshape(-1)

        # ---- diagnostics
        def _nan_count(a): return int(np.isnan(a).sum()) if a is not None else -1
        def _finite_count(a): return int(np.isfinite(a).sum()) if a is not None else -1
        console.log(
            f"[diag] eval_source={eval_source} "
            f"shape(vals)={vals.shape} shape(evals)={(evals.shape if evals is not None else None)} "
            f"nan_evals={_nan_count(evals)} finite_evals={_finite_count(evals)} "
            f"min_eval={(np.nanmin(evals) if evals is not None and evals.size else np.nan):.4f} "
            f"max_eval={(np.nanmax(evals) if evals is not None and evals.size else np.nan):.4f}"
        )

        # ---- determine generation elite
        gen_best_theta = None
        gen_best_fit = -float("inf")

        if (evals is not None) and (evals.size == len(vals)) and np.isfinite(evals).any():
            i = int(np.nanargmax(evals))
            gen_best_fit = float(evals[i])
            gen_best_theta = vals[i].copy()
        elif theta_pb is not None and np.isfinite(fit_pb):
            gen_best_fit = fit_pb
            gen_best_theta = theta_pb.copy()
        else:
            raise RuntimeError("Could not determine generation elite (evals and status both invalid).")

        # ---- update global best + print summary
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_theta = gen_best_theta
            _, _, best_diags = problem._rollout_fitness(best_theta)

        mean_fit = float(np.nanmean(evals)) if (evals is not None and evals.size) else float("nan")
        A = float(best_theta[nu]) if best_theta is not None else float("nan")
        f = float(best_theta[nu + 1]) if best_theta is not None else float("nan")
        phase_std = float(np.std(best_theta[:nu])) if best_theta is not None else float("nan")
        clamp_ratio = (best_diags or {}).get("clamp_ratio", float("nan"))
        dist = (best_diags or {}).get("distance", float("nan"))
        rolled_time = (best_diags or {}).get("rolled_time_s", float("nan"))

        console.log(
            f"[gen {gen:02d}] best_fit={best_fit:.4f}  mean_fit={mean_fit:.4f}  "
            f"A={A:.3f}  f={f:.3f}  phase_std={phase_std:.3f}  "
            f"clamp_ratio={clamp_ratio:.3f}  dist={dist:.3f}  rolled_time={rolled_time:.2f}s"
        )

    console.rule(f"[bold cyan]Training done. Best fitness: {best_fit:.4f}[/bold cyan]")

    if best_theta is None:
        raise RuntimeError("best_theta is None after training; elite extraction failed.")

    # --- Final replay (no early-stop): set θ*, record video, then viewer ---
    # Temporarily disable early stop for replay
    global EARLY_STOP
    EARLY_STOP = False

    phases = best_theta[:nu]
    A = float(best_theta[nu])
    f = float(best_theta[nu + 1])
    omega = 2.0 * math.pi * f
    with torch.inference_mode():
        problem.cpg.phase[:] = torch.from_numpy(phases.astype(np.float32))
        problem.cpg.amplitudes[:] = A
        problem.cpg.w[:] = omega

    problem.tracker.reset()
    mj.mj_resetData(problem.model, problem.data)

    # Record exact-best video (Ariel utilities; plain renderer to avoid tracking issues)
    video_folder = str(DATA / "videos")
    vr = VideoRecorder(output_folder=video_folder)
    video_renderer(
        problem.model,
        problem.data,
        duration=int(SIM_SECONDS),
        video_recorder=vr,
    )
    console.log(f"[green]Saved video(s) to {video_folder}[/green]")

    # Launch interactive viewer with the best θ*
    console.log("[magenta]Launching viewer… close the window to finish.[/magenta]")
    viewer.launch(model=problem.model, data=problem.data)
    mj.set_mjcb_control(None)

    # Save trajectory CSV from the replay
    traj = problem._extract_traj()
    out_csv = DATA / "best_traj.csv"
    with open(out_csv, "w") as f:
        f.write("x,y,z\n")
        for (x, y, z) in traj:
            f.write(f"{x},{y},{z}\n")
    console.log(f"[green]Saved best trajectory to {out_csv}[/green]")


if __name__ == "__main__":
    main()
