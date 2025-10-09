"""Assignment 3 – Minimal GA + export of best robot video and JSON."""

# ---------- Imports ----------
from pathlib import Path
from typing import TYPE_CHECKING, Any
import mujoco as mj
import numpy as np
import numpy.typing as npt

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
POP_SIZE = 10
N_GEN = 5

# ---------- Controller ----------
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
def experiment(robot: Any, controller: Controller, duration: int = 8, record: bool = False) -> None:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
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
    return [
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
    ]

def crossover(g1: list[np.ndarray], g2: list[np.ndarray]) -> list[np.ndarray]:
    child = []
    for v1, v2 in zip(g1, g2):
        cx = RNG.integers(1, GENOTYPE_SIZE - 1)
        new_v = np.concatenate([v1[:cx], v2[cx:]])
        child.append(new_v)
    return child

def mutate(geno: list[np.ndarray], pm: float = 0.1, sigma: float = 0.1) -> list[np.ndarray]:
    new = []
    for v in geno:
        mask = RNG.random(v.shape) < pm
        noise = RNG.normal(0, sigma, size=v.shape)
        mutated = np.clip(v + mask * noise, 0.0, 1.0)
        new.append(mutated.astype(np.float32))
    return new

# ---------- Evaluation ----------
def evaluate(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray]) -> tuple[float, "DiGraph"]:
    """Decode → build → simulate → fitness."""
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    experiment(robot=core, controller=ctrl, duration=6)
    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return fit, robot_graph

# ---------- GA main ----------
def main() -> None:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    console.log(f"[bold cyan]Starting minimal GA for {N_GEN} generations, pop={POP_SIZE}[/bold cyan]")

    population = [random_genotype() for _ in range(POP_SIZE)]
    best_fit = -np.inf
    best_geno = None
    best_graph = None

    for gen in range(N_GEN):
        console.rule(f"[bold green]Generation {gen}[/bold green]")
        fitnesses = np.zeros(POP_SIZE)

        for i, geno in enumerate(population):
            fit, graph = evaluate(nde, geno)
            fitnesses[i] = fit
            console.log(f"Robot {i:02d} → fitness = {fit:.4f}")
            if fit > best_fit:
                best_fit, best_geno, best_graph = fit, geno, graph

        elite_idx = np.argsort(fitnesses)[-2:]
        elites = [population[i] for i in elite_idx]
        console.log(f"Elites: {elite_idx}, best fitness={fitnesses[elite_idx[-1]]:.4f}")

        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = RNG.choice(elites, 2, replace=True)
            child = crossover(p1, p2)
            child = mutate(child, pm=0.1, sigma=0.08)
            new_pop.append(child)
        population = new_pop

    # ---------- After final generation ----------
    console.rule("[bold magenta]Final best robot[/bold magenta]")
    console.log(f"Best fitness = {best_fit:.4f}")

    # Save graph JSON
    out_dir = DATA / "A3_final_best"
    out_dir.mkdir(exist_ok=True)
    save_graph_as_json(best_graph, out_dir / "best_robot_graph.json")

    # Rebuild model for video
    best_core = construct_mjspec_from_graph(best_graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    console.log("[yellow]Recording video of best robot...[/yellow]")
    experiment(robot=best_core, controller=ctrl, duration=10, record=True)

    console.log(f"[green]All done! Video and graph saved.[/green], at {out_dir}")

if __name__ == "__main__":
    main()
