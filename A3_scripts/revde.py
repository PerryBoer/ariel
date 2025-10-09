"""Assignment 3 – Minimal GA + per-body CPG (ES) + export of best robot video and JSON."""

# ---------- Imports ----------
from pathlib import Path
from typing import TYPE_CHECKING, Any
import math
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

# ---- inner CPG (per body) small-ES settings ----
INNER_POP = 12         # number of controller candidates per generation
INNER_GENS = 6         # inner iterations
AMP_MIN, AMP_MAX = 0.0, 0.8
FREQ_MIN, FREQ_MAX = 0.4, 1.4
ALPHA = 0.12           # smoothing toward target
ES_SIGMA_FRAC = 0.25   # initial sigma as a fraction of (hi - lo)
ES_KEEP = 4            # μ best to recombine
ES_MUT_FRAC = 0.15     # mutation noise added to new mean each gen

# ---- GA exploration knobs ----
ELITE_K = 2
IMM_FRAC = 0.25        # fraction of next population as fresh random immigrants
MUT_P = 0.20           # per-gene mutation probability
MUT_SIGMA = 0.15       # mutation std

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

# ---------- CPG utilities ----------
def make_cpg_action_fn(params: npt.NDArray[np.float64], model: mj.MjModel) -> callable:
    """
    params = [phase_0..phase_{nu-1}, AMP, FREQ]
    Returns (m, d) -> control array within actuator ranges.
    """
    nu = int(model.nu)
    phases = np.asarray(params[:nu], dtype=np.float64)
    A = float(params[nu])
    F = float(params[nu + 1])

    phases = np.clip(phases, -math.pi, math.pi)
    A = float(np.clip(A, AMP_MIN, AMP_MAX))
    F = float(np.clip(F, FREQ_MIN, FREQ_MAX))

    def act(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        if m.nu == 0:
            return np.zeros(0, dtype=np.float64)
        lo = m.actuator_ctrlrange[:, 0]
        hi = m.actuator_ctrlrange[:, 1]
        center = 0.5 * (hi + lo)
        half_span = 0.5 * (hi - lo)
        t = d.time
        y = A * np.sin(2.0 * math.pi * F * t + phases)  # [-A, A]
        y = np.clip(y, -1.0, 1.0)
        target = center + half_span * y
        # optional smoothing toward target
        d.ctrl[:] = d.ctrl + ALPHA * (target - d.ctrl)
        return d.ctrl.copy()
    return act

def eval_cpg_params_on_graph(graph: "DiGraph", params: npt.NDArray[np.float64], seconds: int = 6) -> float:
    core = construct_mjspec_from_graph(graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # we'll build model inside Controller callback, as per template
    def controller_cb(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        return make_cpg_action_fn(params, model)(model, data)

    ctrl = Controller(controller_callback_function=controller_cb, tracker=tracker)
    experiment(robot=core, controller=ctrl, duration=seconds, record=False)
    hist = tracker.history["xpos"][0]
    return fitness_function(hist)

def inner_es_optimize_cpg(graph: "DiGraph") -> npt.NDArray[np.float64]:
    """
    Minimal isotropic ES/CMA-like inner optimization for phases + A + F.
    Keeps everything template-compatible and dependency-free.
    """
    # probe ν (number of actuators)
    core = construct_mjspec_from_graph(graph)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS, correct_for_bounding_box=False)
    model = world.spec.compile()
    nu = int(model.nu)

    L = nu + 2
    lo = np.concatenate([np.full(nu, -math.pi), [AMP_MIN], [FREQ_MIN]])
    hi = np.concatenate([np.full(nu,  math.pi), [AMP_MAX], [FREQ_MAX]])
    span = hi - lo

    # sensible center: phases=0, A=0.4, F=1.0
    mean = np.concatenate([np.zeros(nu), [0.4], [1.0]]).astype(np.float64)
    sigma = ES_SIGMA_FRAC * span

    # initialize population
    pop = INNER_POP
    keep = min(ES_KEEP, pop // 2)

    for g in range(INNER_GENS):
        # sample λ candidates
        Z = RNG.normal(0.0, 1.0, size=(pop, L))
        X = mean + sigma * Z
        X = np.clip(X, lo, hi)

        # evaluate
        fits = np.empty(pop, dtype=np.float64)
        for i in range(pop):
            fits[i] = eval_cpg_params_on_graph(graph, X[i], seconds=4)

        # select μ best and recombine (weighted mean)
        idx = np.argsort(fits)[-keep:]  # maximizing fitness (less negative is better)
        elites = X[idx]
        w = np.linspace(1.0, 2.0, keep)  # simple increasing weights
        w /= w.sum()
        new_mean = (w[:, None] * elites).sum(axis=0)

        # small mutation jitter on mean to keep exploration
        mean = np.clip(new_mean + RNG.normal(0.0, ES_MUT_FRAC, size=L) * sigma, lo, hi)

        # optional sigma damp (very light)
        sigma = np.maximum(0.9 * sigma, 1e-3 * span)

    return mean

# ---------- Genotype operations ----------
def random_genotype() -> list[np.ndarray]:
    return [
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
        RNG.random(GENOTYPE_SIZE).astype(np.float32),
    ]

def uniform_crossover_vec(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    mask = RNG.random(v1.shape) < 0.5
    return np.where(mask, v1, v2)

def blend_crossover_vec(v1: np.ndarray, v2: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    # arithmetic blend in an extended interval around parents
    lo = np.minimum(v1, v2) - alpha * np.abs(v1 - v2)
    hi = np.maximum(v1, v2) + alpha * np.abs(v1 - v2)
    child = RNG.uniform(lo, hi)
    return np.clip(child, 0.0, 1.0)

def crossover(g1: list[np.ndarray], g2: list[np.ndarray]) -> list[np.ndarray]:
    child = []
    for v1, v2 in zip(g1, g2):
        if RNG.random() < 0.5:
            new_v = uniform_crossover_vec(v1, v2)
        else:
            new_v = blend_crossover_vec(v1, v2)
        child.append(new_v.astype(np.float32))
    return child

def mutate(geno: list[np.ndarray], pm: float = MUT_P, sigma: float = MUT_SIGMA) -> list[np.ndarray]:
    new = []
    for v in geno:
        mask = RNG.random(v.shape) < pm
        noise = RNG.normal(0, sigma, size=v.shape)
        mutated = np.clip(v + mask * noise, 0.0, 1.0)
        new.append(mutated.astype(np.float32))
    return new

# ---------- Evaluation ----------
def evaluate(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray]) -> tuple[float, "DiGraph"]:
    """
    Decode → build → (inner ES to tune CPG for this body) → simulate with best CPG → fitness.
    """
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)

    # inner ES optimize a CPG for THIS body
    theta = inner_es_optimize_cpg(robot_graph)

    # final evaluation with the tuned controller
    core = construct_mjspec_from_graph(robot_graph)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    def controller_cb(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        return make_cpg_action_fn(theta, model)(model, data)

    ctrl = Controller(controller_callback_function=controller_cb, tracker=tracker)
    experiment(robot=core, controller=ctrl, duration=6)
    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return fit, robot_graph

# ---------- GA main ----------
def main() -> None:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    console.log(f"[bold cyan]Starting GA for {N_GEN} generations, pop={POP_SIZE} (inner ES: pop={INNER_POP}, gens={INNER_GENS})[/bold cyan]")

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

        # --- selection (elites + offspring + immigrants) ---
        elite_idx = np.argsort(fitnesses)[-ELITE_K:]
        elites = [population[i] for i in elite_idx]
        console.log(f"Elites: {elite_idx}, best fitness={fitnesses[elite_idx[-1]]:.4f}")

        next_pop = elites.copy()

        # random immigrants for exploration
        n_imm = max(1, int(IMM_FRAC * POP_SIZE))
        for _ in range(n_imm):
            next_pop.append(random_genotype())

        # fill the rest with offspring from elites and random parents
        while len(next_pop) < POP_SIZE:
            if RNG.random() < 0.5:
                p1, p2 = RNG.choice(elites, 2, replace=True)
            else:
                # occasionally breed with a fresh random to increase diversity
                p1 = RNG.choice(elites)
                p2 = random_genotype()
            child = mutate(crossover(p1, p2))
            next_pop.append(child)

        population = next_pop[:POP_SIZE]

    # ---------- After final generation ----------
    console.rule("[bold magenta]Final best robot[/bold magenta]")
    console.log(f"Best fitness = {best_fit:.4f}")

    out_dir = DATA / "A3_final_best"
    out_dir.mkdir(exist_ok=True)
    save_graph_as_json(best_graph, out_dir / "best_robot_graph.json")

    # Rebuild model for video with tuned CPG again (short ES to get params)
    best_core = construct_mjspec_from_graph(best_graph)
    theta = inner_es_optimize_cpg(best_graph)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    def controller_cb(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        return make_cpg_action_fn(theta, model)(model, data)

    ctrl = Controller(controller_callback_function=controller_cb, tracker=tracker)
    console.log("[yellow]Recording video of best robot...[/yellow]")
    experiment(robot=best_core, controller=ctrl, duration=10, record=True)

    console.log(f"[green]All done! Video and graph saved.[/green], at {out_dir}")

if __name__ == "__main__":
    main()
