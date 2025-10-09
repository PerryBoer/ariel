"""A3 — Baseline GA with template-style random NN controller (no CPG yet).

What it does:
- Evolves ONLY the 3 NDE input vectors (type, conn, rot)
- Uses a random NN controller PER evaluation that RETURNS outputs (template contract)
- Prints each individual’s fitness; selects top 25% (no elitism); crossover+mutation to refill
- Records a VIDEO whenever we find a new global best (template behavior)
- Plots the XY trajectory of the best-of-run body (like the template)
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
import math
import numpy as np
import numpy.typing as npt
import mujoco as mj
import matplotlib.pyplot as plt

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    CoreModule,
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import (
    single_frame_renderer,
    tracking_video_renderer,
    video_renderer,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from networkx import DiGraph

# ----------------------------- Config -----------------------------
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
(DATA / "videos").mkdir(exist_ok=True)

# Arena & target
SPAWN_POS = [-0.8, 0.0, 0.15]
TARGET_POSITION = np.array([5.0, 0.0, 0.5], dtype=np.float64)

# Morphology / genotype
NUM_OF_MODULES = 30
GENOTYPE_SIZE = 64

# GA
POP_SIZE = 16
N_GEN = 6
SELECT_FRAC = 0.25
PM = 0.10
SIGMA = 0.08

# Simulation
DURATION_SEC = 8

# Viewer types for parity with template
ViewerTypes = Literal["launcher", "video", "simple", "tracking", "no_control", "frame"]


# ----------------------------- Helpers -----------------------------
def fitness_function(history: list[list[float]]) -> float:
    """Negative 3D distance to target at the end (maximize)."""
    if not history:
        return -1e9
    last = np.array(history[-1], dtype=np.float64)
    return -float(np.linalg.norm(TARGET_POSITION - last))


def show_xpos_history(history: list[list[float]]) -> None:
    """Recreates the template-style XY path plot over a background frame."""
    # Set up a free camera to make a background still
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # World background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, save_path=save_path, save=True)

    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    pos = np.array(history, dtype=np.float64)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]
    pixel_to_dist = -((ymc - ym0) / (yc - y0))

    pix = [[xc, yc]]
    for i in range(len(pos) - 1):
        xi, yi, _ = pos[i]
        xj, yj, _ = pos[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pix[i]
        pix.append([xn + int(xd), yn + int(yd)])
    pix = np.array(pix)

    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pix[:, 0], pix[:, 1], "b-", label="Path")
    ax.plot(pix[-1, 0], pix[-1, 1], "ro", label="End")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.savefig(DATA / "robot_path.png")


# ----------------------- Template-style controller -----------------------
def nn_controller_factory(model: mj.MjModel) -> callable:
    """Return a callback(model, data)->outputs vector. Smoothing is done by Controller.set_control."""
    nu = int(model.nu)
    if nu == 0:
        # Return a harmless callback that outputs empty array
        def empty_cb(_m: mj.MjModel, _d: mj.MjData) -> npt.NDArray[np.float64]:
            return np.zeros(0, dtype=np.float64)
        return empty_cb

    # Sample random weights ONCE per evaluation
    in_dim = len(mj.MjData(model).qpos)
    h = 8
    W1 = RNG.normal(0.0, 0.35, size=(in_dim, h)).astype(np.float64)
    W2 = RNG.normal(0.0, 0.35, size=(h, nu)).astype(np.float64)

    # Compute actuator-safe limits intersected with ±π/2
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    lo = np.maximum(lo, -math.pi / 2.0)
    hi = np.minimum(hi,  math.pi / 2.0)

    def cb(_m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        x = d.qpos
        h1 = np.tanh(x @ W1)
        out = np.tanh(h1 @ W2) * (math.pi / 2.0)
        # return the output; Controller will smooth and clip
        np.clip(out, lo, hi, out=out)
        return out

    return cb


# --------------------- Build body from genotype ---------------------
def build_robot_from_genotype(
    nde: NeuralDevelopmentalEncoding,
    genotype: list[np.ndarray],
) -> tuple[CoreModule, "DiGraph"]:
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    return construct_mjspec_from_graph(graph), graph


# ----------------------------- Simulation -----------------------------
def run_episode(
    robot: CoreModule,
    controller_cb: callable,
    *,
    duration: int = DURATION_SEC,
    mode: ViewerTypes = "simple",
    track: bool = True,
    record_path: Path | None = None,
) -> tuple[list[list[float]], float]:
    """Runs one episode and returns (xpos_history, fitness)."""
    # Fresh world/model/data
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Initialize ctrl to zeros to avoid None in first smoothing step
    if model.nu > 0:
        data.ctrl[:] = 0.0

    # Tracker
    tracker = None
    if track:
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        tracker.setup(world.spec, data)

    # Template Controller: it expects the callback to RETURN outputs
    controller = Controller(controller_callback_function=controller_cb, tracker=tracker)

    # Set control callback
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    # Run with chosen mode
    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "video":
            recorder = VideoRecorder(output_folder=str(record_path or (DATA / "videos")))
            video_renderer(model, data, duration=duration, video_recorder=recorder)
        case "tracking":
            recorder = VideoRecorder(output_folder=str(record_path or (DATA / "videos")))
            tracking_video_renderer(model, data, duration=duration, video_recorder=recorder)
        case "launcher":
            from mujoco import viewer
            viewer.launch(model=model, data=data)
        case "no_control":
            mj.set_mjcb_control(None)
            from mujoco import viewer
            viewer.launch(model=model, data=data)
        case "frame":
            single_frame_renderer(model, data, save=True, save_path=str(DATA / "frame.png"))
        case _:
            simple_runner(model, data, duration=duration)

    # Read history & fitness
    xpos_hist = tracker.history["xpos"][0] if tracker is not None else [[float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2] if model.nq >= 3 else 0.0)]]
    fit = fitness_function(xpos_hist)
    # Clear callback after episode
    mj.set_mjcb_control(None)
    return xpos_hist, fit


# ----------------------------- GA operators -----------------------------
def random_genotype() -> list[np.ndarray]:
    return [
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
    ]


def crossover(g1: list[np.ndarray], g2: list[np.ndarray]) -> list[np.ndarray]:
    child: list[np.ndarray] = []
    for v1, v2 in zip(g1, g2):
        cx = RNG.integers(1, GENOTYPE_SIZE - 1)
        child.append(np.concatenate([v1[:cx], v2[cx:]]).astype(np.float32))
    return child


def mutate(g: list[np.ndarray], pm: float = PM, sigma: float = SIGMA) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for v in g:
        mask = RNG.random(v.shape) < pm
        noise = RNG.normal(0.0, sigma, size=v.shape).astype(np.float32)
        vv = v + mask * noise
        out.append(np.clip(vv, 0.0, 1.0).astype(np.float32))
    return out


# ----------------------------- Main GA -----------------------------
def main() -> None:
    # One NDE instance for deterministic mapping (TA tip)
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)

    # Init population
    population: list[list[np.ndarray]] = [random_genotype() for _ in range(POP_SIZE)]

    best_fit = -np.inf
    best_hist: list[list[float]] | None = None
    best_graph: "DiGraph" | None = None
    best_robot: CoreModule | None = None

    console.log(f"[bold cyan]Baseline GA started[/bold cyan]  pop={POP_SIZE}, G={N_GEN}, select={int(SELECT_FRAC*100)}%  (random NN controller; no elitism)")

    for gen in range(N_GEN):
        console.rule(f"[bold green]Generation {gen}[/bold green]")

        fits = np.empty(POP_SIZE, dtype=np.float64)

        # Evaluate
        for i, geno in enumerate(population):
            robot, graph = build_robot_from_genotype(nde, geno)
            # controller for THIS body/model (returns outputs)
            # we need a compiled model to size the NN → use run_episode which compiles inside
            # run once in simple mode, track path & fitness
            # The episode will build the NN from the model it compiles
            # So we pass a factory which will be created inside run_episode
            # Simplest: compile a temp model to build the NN now:
            mj.set_mjcb_control(None)
            tmp_world = OlympicArena()
            tmp_world.spawn(robot.spec, position=SPAWN_POS)
            tmp_model = tmp_world.spec.compile()
            ctrl_cb = nn_controller_factory(tmp_model)
            # evaluate
            hist, fit = run_episode(robot, ctrl_cb, duration=DURATION_SEC, mode="simple", track=True)
            fits[i] = fit
            console.log(f"[dim]Gen {gen:02d}[/dim]  Ind {i:02d}  fitness = {fit:+.4f}")

            if fit > best_fit:
                best_fit = fit
                best_hist = hist
                best_graph = graph
                best_robot = robot


        # Generation summary
        console.log(f"[yellow]Gen {gen} summary:[/yellow]  best={fits.max():+.4f}  mean={fits.mean():+.4f}  median={np.median(fits):+.4f}")

        # Selection (top 25% as mating pool; NO elitism)
        k = max(1, int(math.ceil(SELECT_FRAC * POP_SIZE)))
        parent_idx = np.argsort(fits)[-k:]
        parents = [population[j] for j in parent_idx]

        # Refill population with children only
        new_pop: list[list[np.ndarray]] = []
        while len(new_pop) < POP_SIZE:
            p1, p2 = RNG.choice(parents, size=2, replace=True)
            child = mutate(crossover(p1, p2), pm=PM, sigma=SIGMA)
            new_pop.append(child)
        population = new_pop

    console.rule("[bold magenta]Done[/bold magenta]")
    console.log(f"[green]Best overall fitness[/green] = {best_fit:+.4f}")

    console.log("[yellow]global best → recording video[/yellow]")
    _ = run_episode(robot, ctrl_cb, duration=max(DURATION_SEC, 10), mode="video", track=True, record_path=DATA / "videos")
    console.log(f"Saved best video → {DATA / 'videos'}")

    # Save best robot JSON and XY plot
    if best_graph is not None and best_hist is not None:
        save_graph_as_json(best_graph, DATA / "best_robot_graph.json")
        console.log(f"Saved best robot graph → {DATA / 'best_robot_graph.json'}")
        show_xpos_history(best_hist)
        console.log(f"Saved best XY path plot → {DATA / 'robot_path.png'}")


if __name__ == "__main__":
    main()
