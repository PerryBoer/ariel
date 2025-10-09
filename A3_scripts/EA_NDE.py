"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import math
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
import torch
from mujoco import viewer

# EvoTorch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.core import SolutionBatch
from evotorch.logging import StdOutLogger

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
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

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.show()


# =========================
# CPG + CMA-ES ADDITIONS
# =========================

# Bounds & smoothing
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
AMP_MIN, AMP_MAX = 0.0, 0.8
FREQ_MIN, FREQ_MAX = 0.4, 1.4
ALPHA = 0.08  # gentler smoothing for stability


def make_cpg_controller_from_params(
    params: np.ndarray,
    model: mj.MjModel,
) -> callable:
    """
    params = [phase_0..phase_{nu-1}, AMP, FREQ]
    Produces a MuJoCo control callback: (model, data) -> np.ndarray
    """
    nu = int(model.nu)
    if nu == 0:
        def empty_cb(m, d):
            return np.zeros(0, dtype=np.float64)
        return empty_cb

    phases = np.asarray(params[:nu], dtype=np.float64)
    AMP = float(params[nu])
    FREQ = float(params[nu + 1])

    # Safety clipping
    phases = np.clip(phases, PHASE_MIN, PHASE_MAX)
    AMP = float(np.clip(AMP, AMP_MIN, AMP_MAX))
    FREQ = float(np.clip(FREQ, FREQ_MIN, FREQ_MAX))  # correct bounds

    lo = model.actuator_ctrlrange[:, 0] if nu > 0 else np.array([])
    hi = model.actuator_ctrlrange[:, 1] if nu > 0 else np.array([])
    center = 0.5 * (hi + lo) if nu > 0 else np.array([])
    half_span = 0.5 * (hi - lo) if nu > 0 else np.array([])

    def cb(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        if m.nu == 0:
            return np.zeros(0, dtype=np.float64)
        t = d.time
        y = AMP * np.sin(2.0 * math.pi * FREQ * t + phases)  # [-AMP, AMP]
        y = np.clip(y, -1.0, 1.0)
        target = center + half_span * y
        np.clip(target, lo, hi, out=target)  # be safe in range
        if d.ctrl is not None and len(d.ctrl) == m.nu:
            d.ctrl[:] = d.ctrl + ALPHA * (target - d.ctrl)
            return d.ctrl
        else:
            return target.astype(np.float64)

    return cb


def rollout_fitness_with_params(
    robot_core: Any,
    params: np.ndarray,
    seconds: float = 1.5,  # short & cheap for CMA ranking
) -> float:
    """
    Headless inner evaluation used by CMA-ES.
    Reuses the same env (OlympicArena), spawn pos, tracker binding, and fitness_function.
    """
    mj.set_mjcb_control(None)
    world = OlympicArena()

    # Disable bbox correction to avoid compiling child spec directly
    world.spawn(robot_core.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=False)

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if data.ctrl is not None:
        data.ctrl[:] = 0.0

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    controller_cb = make_cpg_controller_from_params(params, model)
    ctrl = Controller(controller_callback_function=controller_cb, tracker=tracker)

    try:
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
        simple_runner(model, data, duration=seconds)

        hist_list = tracker.history.get("xpos", [])
        if not hist_list or len(hist_list[0]) == 0:
            return -1e8  # empty trace → finite but poor

        fit = float(fitness_function(hist_list[0]))
        return fit if np.isfinite(fit) else -1e8

    except Exception as e:
        console.log(f"[red]Rollout failed: {e}[/red]")
        return -1e9
    finally:
        mj.set_mjcb_control(None)


class CPGProblem(Problem):
    """
    EvoTorch Problem: maximize the template's fitness_function using
    CPG params theta = [phases (nu), A, F] for a fixed compiled body (robot_core).
    """
    def __init__(self, robot_core: Any):
        model = robot_core.spec.compile()  # compile once to read nu
        nu = int(model.nu)
        L = nu + 2  # phases + A + F

        lo = np.concatenate(
            [np.full(nu, PHASE_MIN), [AMP_MIN], [FREQ_MIN]]
        ).astype(np.float64)
        hi = np.concatenate(
            [np.full(nu, PHASE_MAX), [AMP_MAX], [FREQ_MAX]]
        ).astype(np.float64)

        self._robot_core = robot_core

        super().__init__(
            objective_sense="max",
            solution_length=L,
            dtype=torch.float64,
            device="cpu",
            initial_bounds=(torch.from_numpy(lo), torch.from_numpy(hi)),
        )

    def evaluate(self, X):
        if isinstance(X, SolutionBatch):
            vals = X.access_values()
            fits = []
            for row in vals:
                theta = row.detach().cpu().numpy()
                f = rollout_fitness_with_params(self._robot_core, theta, seconds=1.5)
                if not np.isfinite(f):
                    f = -1e9
                fits.append(f)
            fits_t = torch.as_tensor(fits, dtype=vals.dtype, device=vals.device)
            X.set_evals(fits_t)
            return fits_t
        elif isinstance(X, torch.Tensor):
            fits = []
            for row in X:
                f = rollout_fitness_with_params(self._robot_core, row.detach().cpu().numpy(), seconds=1.5)
                fits.append(f if np.isfinite(f) else -1e9)
            return torch.as_tensor(fits, dtype=X.dtype, device=X.device)
        else:
            raise TypeError(f"Unsupported input to evaluate(): {type(X)}")


def optimize_cpg_for_core(
    robot_core: Any,
    popsize: int = 8,     # small, fast proof-of-life; bump later (24)
    gens: int = 4,        # small, fast proof-of-life; bump later (12)
    sigma_init: float = 0.25,
) -> np.ndarray:
    model = robot_core.spec.compile()
    nu = int(model.nu)
    center = np.concatenate([np.zeros(nu), [0.3], [0.9]]).astype(np.float64)

    prob = CPGProblem(robot_core)
    solver = CMAES(
        prob,
        popsize=popsize,
        stdev_init=sigma_init,
        center_init=torch.from_numpy(center),
    )
    _ = StdOutLogger(solver, interval=1)  # EvoTorch logger

    best_theta = center.copy()
    best_fit = -1e12

    for g in range(gens):
        solver.step()

        pop = solver.population
        vals = pop.access_values().detach().cpu().numpy()
        fits_t = pop.get_evals() if hasattr(pop, "get_evals") else pop.evals
        fits = fits_t.detach().cpu().numpy()
        fits = np.where(np.isfinite(fits), fits, -1e9)

        i = int(np.argmax(fits))
        pop_best = float(fits[i])
        pop_mean = float(np.mean(fits))

        if pop_best > best_fit:
            best_fit = pop_best
            best_theta = vals[i].copy()

        # explicit per-gen progress
        console.log(f"[blue]gen {g:02d}[/blue] best={pop_best:.4f}  mean={pop_mean:.4f}")

    console.log(f"[yellow]CMA-ES best inner rollout fitness: {best_fit:.4f}[/yellow]")
    return best_theta


# (Original nn_controller kept for reference but unused in final run)
def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))
    inputs = data.qpos
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world (outer run can use default bbox correction)
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    args: list[Any] = []
    kwargs: dict[Any, Any] = {}

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "frame":
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        case "launcher":
            viewer.launch(model=model, data=data)
        case "no_control":
            mj.set_mjcb_control(None)
            viewer.launch(model=model, data=data)
    # ==================================================================== #


def main() -> None:
    """Entry point."""
    # --- Random genotype for NDE ---
    genotype_size = 64
    genotype = [
        RNG.random(genotype_size).astype(np.float32),  # type_p_genes
        RNG.random(genotype_size).astype(np.float32),  # conn_p_genes
        RNG.random(genotype_size).astype(np.float32),  # rot_p_genes
    ]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the graph to a file
    save_graph_as_json(robot_graph, DATA / "robot_graph.json")

    # Build compiled core
    core = construct_mjspec_from_graph(robot_graph)

    # === Train a CPG controller (CMA-ES) FOR THIS BODY ===
    console.log("[yellow]Optimizing CPG with CMA-ES for this body...[/yellow]")
    theta = optimize_cpg_for_core(core, popsize=8, gens=4, sigma_init=0.25)  # bump later

    def cpg_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        return make_cpg_controller_from_params(theta, model)(model, data)

    # Tracker (unchanged)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

    # Simulate the robot with the optimized CPG
    ctrl = Controller(controller_callback_function=cpg_controller, tracker=tracker)

    video_dir = DATA / "videos"
    console.log(f"[cyan]Saving video to: {video_dir}[/cyan]")
    experiment(robot=core, controller=ctrl, mode="video", duration=10)
    console.log("[green]Video render complete.[/green]")

    # Plot & fitness if we recorded a trajectory
    hist_ok = tracker.history.get("xpos") and len(tracker.history["xpos"][0]) > 1
    if hist_ok:
        show_xpos_history(tracker.history["xpos"][0])
        fitness = fitness_function(tracker.history["xpos"][0])
        console.log(f"[green]Fitness of CPG-optimized robot: {fitness:.4f}[/green]")
    else:
        console.log("[yellow]No recorded trajectory to plot.[/yellow]")


if __name__ == "__main__":
    main()
