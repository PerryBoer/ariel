# a3_single_nacpg_run.py
# One-body test with a Normalized Asymmetric CPG controller (template-compatible).

from __future__ import annotations

# Standard libs
from pathlib import Path
import math
import numpy as np
import numpy.typing as npt

# Third-party
import mujoco as mj
import torch
from torch import nn
from rich.console import Console
from rich.traceback import install
from mujoco import viewer

# Ariel (template pieces)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.tracker import Tracker  # <-- needed tracker

# ------------------------ Console / Torch setup ------------------------
install(show_locals=False)
console = Console()
torch.set_printoptions(precision=4)

# ----------------------------- Config ---------------------------------
SEED = 42
RNG = np.random.default_rng(SEED)
NUM_OF_MODULES = 30
GENOTYPE_SIZE = 64
SPAWN_POS = [-0.8, 0.0, 0.15]

OUTDIR = Path("__data__/single_nacpg")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------ Helper: adjacency ----------------------------
def create_fully_connected_adjacency(num_nodes: int) -> dict[int, list[int]]:
    return {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}

# ================== Normalized Asymmetric CPG (renamed) ==================
class NormalizedAsymmetricCPGController(nn.Module):
    """
    A Normalized Asymmetric CPG (NA-CPG) turned into a template-compatible controller.
    - Size is tied to model.nu (one oscillator per actuator).
    - forward() advances oscillator states; mj_callback() returns actuator targets.
    - No smoothing here; the template's Controller blends & clips each step.
    """

    # buffers (for type hints)
    xy: torch.Tensor
    xy_dot_old: torch.Tensor
    angles: torch.Tensor

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 0.1,
        dt: float = 0.01,
        hard_bounds: tuple[float, float] | None = (-torch.pi / 2, torch.pi / 2),
        *,
        angle_tracking: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        # --- user params
        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        self.angle_tracking = angle_tracking
        self.hard_bounds = hard_bounds
        self.clamping_error = 0.0
        if seed is not None:
            torch.manual_seed(seed)

        # --- inherent (fixed during learning)
        self.alpha = alpha            # learning/flow rate
        self.dt = dt                  # time step for integrator

        # --- adaptable parameters (can be optimized later)
        scale = torch.pi * 2
        # definitely adapt
        self.phase = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        self.amplitudes = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        # probably adapt
        self.w = nn.Parameter(((torch.rand(self.n) * 2 - 1) * scale), requires_grad=False)
        # probably not to adapt (but kept open)
        self.ha = nn.Parameter(torch.randn(self.n), requires_grad=False)
        self.b = nn.Parameter(
            torch.randint(-100, 100, (self.n,), dtype=torch.float32),
            requires_grad=False,
        )
        self.parameter_groups = {
            "phase": self.phase,
            "w": self.w,
            "amplitudes": self.amplitudes,
            "ha": self.ha,
            "b": self.b,
        }
        self.num_of_parameters = sum(p.numel() for p in self.parameters())
        self.num_of_parameter_groups = len(self.parameter_groups)

        # --- internal state (buffers)
        self.register_buffer("xy", torch.randn(self.n, 2))
        self.register_buffer("xy_dot_old", torch.randn(self.n, 2))
        self.register_buffer("angles", torch.zeros(self.n))
        self.angle_history: list[list[float]] = []
        self.initial_state = {
            "xy": self.xy.clone(),
            "xy_dot_old": self.xy_dot_old.clone(),
            "angles": self.angles.clone(),
        }

        # range caches (numpy + torch, filled by _bind_model_ranges)
        self._ctrl_lo_np: np.ndarray | None = None
        self._ctrl_hi_np: np.ndarray | None = None
        self._center_np:   np.ndarray | None = None
        self._half_span_np:np.ndarray | None = None

    # ---------------- param helpers (unchanged, handy for later training) -------------
    def param_type_converter(self, params: list[float] | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(params, list):
            params = torch.tensor(params, dtype=torch.float32)
        elif isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        return params

    def set_flat_params(self, params: torch.Tensor) -> None:
        safe_params = self.param_type_converter(params)
        if safe_params.numel() != self.num_of_parameters:
            raise ValueError(f"Parameter vector has incorrect size. Expected {self.num_of_parameters}, got {safe_params.numel()}.")
        pointer = 0
        for param in self.parameter_groups.values():
            num_param = param.numel()
            param.data = safe_params[pointer : pointer + num_param].view_as(param)
            pointer += num_param

    def set_param_with_dict(self, params: dict[str, torch.Tensor]) -> None:
        for key, value in params.items():
            self.set_params_by_group(key, self.param_type_converter(value))

    def set_params_by_group(self, group_name: str, params: torch.Tensor) -> None:
        safe_params = self.param_type_converter(params)
        if group_name not in self.parameter_groups:
            raise ValueError(f"Parameter group '{group_name}' does not exist.")
        param = self.parameter_groups[group_name]
        if safe_params.numel() != param.numel():
            raise ValueError(f"Parameter vector has incorrect size for group '{group_name}'.")
        param.data = safe_params.view_as(param)

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameter_groups.values()])

    # ---------------- core oscillator pieces -----------------------------------------
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
        self.xy.data = self.initial_state["xy"].clone()
        self.xy_dot_old.data = self.initial_state["xy_dot_old"].clone()
        self.angles.data = self.initial_state["angles"].clone()
        self.angle_history = []

    def forward(self, time: float | None = None) -> torch.Tensor:
        # reset at episode start
        if time is not None and torch.isclose(torch.tensor(time), torch.tensor(0.0)):
            self.reset()

        with torch.inference_mode():
            # build coupling matrix R (phase diffs)
            r_matrix = torch.zeros(self.n, self.n, 2, 2)
            eye2 = torch.eye(2)
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        r_matrix[i, j] = eye2
                    else:
                        phase_diff_ij = self.phase[i] - self.phase[j]
                        cos_d_ij = torch.cos(phase_diff_ij)
                        sin_d_ij = torch.sin(phase_diff_ij)
                        r_matrix[i, j] = torch.tensor([[cos_d_ij, -sin_d_ij],
                                                       [sin_d_ij,  cos_d_ij]])

            # local dynamics matrix K per oscillator
            k_matrix = torch.zeros(self.n, 2, 2)
            for i in range(self.n):
                x_dot_old, _ = self.xy_dot_old[i]
                ha_i = self.ha[i]
                w_i = self.w[i]
                xi, yi = self.xy[i]

                r2i = xi**2 + yi**2
                term_a = self.term_a(self.alpha, r2i)          # radial contraction term
                zeta_i = self.zeta(ha_i, x_dot_old)
                term_b = self.term_b(zeta_i, w_i)              # angular velocity term

                k_matrix[i] = torch.tensor([[term_a, -term_b],
                                            [term_b,  term_a]])

            # integrate each oscillator
            angles = torch.zeros(self.n)
            for i, (xi, yi) in enumerate(self.xy):
                # local contribution
                term_a_vec = torch.mv(k_matrix[i], self.xy[i])

                # coupling from neighbors
                term_b_vec = torch.zeros(2)
                for j in self.adjacency_dict[i]:
                    term_b_vec += torch.mv(r_matrix[i, j], self.xy[j])

                # derivative
                xi_dot, yi_dot = term_a_vec + term_b_vec

                # constrain rate of change (keeps stability)
                xi_dot_old, yi_dot_old = self.xy_dot_old[i]
                diff = 10.0
                xi_dot = torch.clamp(xi_dot, xi_dot_old - diff, xi_dot_old + diff)
                yi_dot = torch.clamp(yi_dot, yi_dot_old - diff, yi_dot_old + diff)

                # Euler step
                xi_new = xi + (xi_dot * self.dt)
                yi_new = yi + (yi_dot * self.dt)

                # update states
                self.xy_dot_old[i] = self.xy[i]
                self.xy[i] = torch.tensor([xi_new, yi_new])

                # output (angle signal)
                angles[i] = self.amplitudes[i] * yi_new + self.b[i]

            # hard bounds (±π/2) if requested
            if self.hard_bounds is not None:
                pre = angles.clone()
                angles = torch.clamp(angles, min=self.hard_bounds[0], max=self.hard_bounds[1], out=angles)
                self.clamping_error = (pre - angles).abs().sum().item()

            if self.angle_tracking:
                self.angle_history.append(angles.clone().tolist())

        # safety check
        if torch.isnan(angles).any():
            raise ValueError("NaN detected in NA-CPG angles.")
        self.angles = angles
        return self.angles.clone()

    # ---------------- MuJoCo bindings ---------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: mj.MjModel,
        *,
        alpha: float = 0.1,
        seed: int | None = 0,
        angle_tracking: bool = False,
    ) -> "NormalizedAsymmetricCPGController":
        nu = int(model.nu)
        if nu <= 0:
            raise ValueError("Model has zero actuators; NA-CPG is undefined.")
        adj = create_fully_connected_adjacency(nu)
        inst = cls(
            adjacency_dict=adj,
            alpha=alpha,
            dt=float(model.opt.timestep),
            hard_bounds=(-torch.pi / 2, torch.pi / 2),
            angle_tracking=angle_tracking,
            seed=seed,
        )
        # gentle defaults
        with torch.no_grad():
            inst.phase[:]      = 0.0
            inst.amplitudes[:] = 0.5
            inst.w[:]          = 2.0 * torch.pi * 1.0   # ~1 Hz in rad/s
            inst.ha[:]         = 0.25
            inst.b[:]          = 0.0
        inst._bind_model_ranges(model)
        return inst

    def _bind_model_ranges(self, model: mj.MjModel) -> None:
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        lo = np.maximum(lo, -math.pi / 2.0)
        hi = np.minimum(hi,  math.pi / 2.0)
        self._ctrl_lo_np = lo.astype(np.float64)
        self._ctrl_hi_np = hi.astype(np.float64)
        self._center_np    = 0.5 * (self._ctrl_hi_np + self._ctrl_lo_np)
        self._half_span_np = 0.5 * (self._ctrl_hi_np - self._ctrl_lo_np)

        # torch mirrors (optional)
        self.register_buffer("_ctrl_lo", torch.from_numpy(self._ctrl_lo_np))
        self.register_buffer("_ctrl_hi", torch.from_numpy(self._ctrl_hi_np))
        self.register_buffer("_center", torch.from_numpy(self._center_np))
        self.register_buffer("_half_span", torch.from_numpy(self._half_span_np))

    def mj_callback(self, model: mj.MjModel) -> callable:
        """Return a template-compatible callback: (model, data) -> np.ndarray (targets)."""
        if self._ctrl_lo_np is None:
            self._bind_model_ranges(model)

        def _cb(_m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
            if abs(d.time) < 1e-12:
                self.reset()
            angles_t = self.forward(time=float(d.time))    # torch vec (n,)
            angles = angles_t.detach().cpu().numpy()
            # treat angles as final desired target; intersect with actuator range (and ±π/2)
            return np.clip(angles, self._ctrl_lo_np, self._ctrl_hi_np)

        return _cb

# =========================== Build one body =============================
def random_genotype() -> list[np.ndarray]:
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

# =============================== Main ===================================
def main() -> None:
    console.rule("[bold green]A3 — Single Body NA-CPG Run[/bold green]")

    # Deterministic genotype→phenotype mapping (single NDE)
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)

    # Build one random body and save graph for reproducibility
    core, graph = build_random_body(nde)
    save_graph_as_json(graph, OUTDIR / "robot_graph.json")
    console.log(f"Saved robot graph → {OUTDIR / 'robot_graph.json'}")

    # World & model
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if model.nu > 0:
        data.ctrl[:] = 0.0

    # NA-CPG controller (renamed class)
    nacpg = NormalizedAsymmetricCPGController.from_model(model, alpha=0.1, seed=SEED, angle_tracking=False)
    ctrl_cb = nacpg.mj_callback(model)

    # --- FIX: Create a Tracker and wire it to the Controller ---
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    controller = Controller(controller_callback_function=ctrl_cb, tracker=tracker)

    console.log("[bold cyan]Launching viewer… close the window to end the run[/bold cyan]")
    try:
        mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
        viewer.launch(model=model, data=data)  # interactive, blocks until closed
    finally:
        mj.set_mjcb_control(None)

    console.rule("[bold magenta]Done[/bold magenta]")

if __name__ == "__main__":
    main()
