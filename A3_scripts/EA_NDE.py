"""
Evolutionary algorithm to evolve the robot body. Performs:
- roulette wheel parent selection
- one-point crossover per genotype vector
- gaussian mutation per genotype vector
And saves best robot json and video after final generation
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, List, Tuple

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import torch  # <-- added

### Local libraries
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

SEED = 42
RNG = np.random.default_rng(SEED)
np.random.seed(SEED)        # <-- added (HPD may use numpy's global RNG)
torch.manual_seed(SEED)     # <-- added (NDE is a torch module)

### EA parameters
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

# data directory
CWD = Path.cwd()
OUTPUT = CWD / "__output__"
OUTPUT.mkdir(parents=True, exist_ok=True)

# --- instantiate NDE/HPD ONCE per run (minimal change) ---
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
NDE.eval()
for p in NDE.parameters():
    p.requires_grad_(False)
HPD = HighProbabilityDecoder(NUM_MODULES)
# ----------------------------------------------------------

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    """
    Minimal fixed-parameter CPG (global sine) controller.
    - No evolvable controller params (constants only).
    - Deterministic phase offsets per actuator index.
    - Actions clipped to [-pi/2, pi/2] per assignment.
    """
    nu: int = model.nu
    t: float = float(data.time)

    # ---- Fixed CPG constants (NOT evolved) ----
    CONTROL_BOUND = np.pi / 2              # required action bound
    A = 0.6 * CONTROL_BOUND                # amplitude (conservative to avoid saturation)
    FREQ_HZ = 0.7                          # gait frequency in Hz
    OMEGA = 2.0 * np.pi * FREQ_HZ          # rad/s
    BIAS = 0.0                             # center offset

    if nu == 0:
        return np.zeros(0, dtype=np.float64)

    # Evenly spaced phase offsets over actuators (deterministic)
    phases = 2.0 * np.pi * (np.arange(nu, dtype=np.float64) / nu)

    # CPG signal per actuator
    u = A * np.sin(OMEGA * t + phases) + BIAS

    # Enforce assignment-mandated bounds
    np.clip(u, -CONTROL_BOUND, CONTROL_BOUND, out=u)
    return u

### genotype functions
def random_genotype() -> List[np.ndarray]:
    """Initialize random genotype vectors"""
    return [
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
        RNG.random(GENE_LEN).astype(np.float32),
    ]

def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform one-point crossover between two genotype vectors (of the same type)"""
    L = a.size
    point = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1.astype(np.float32), c2.astype(np.float32)

def crossover_per_chromosome(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """From two parents, perform one-point crossover for each of the three vector types"""
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
    """Perform gaussian mutation on a genotype"""
    mutated = []
    for chrom in gen:
        to_mut = RNG.random(chrom.shape) < MUTATION_PROB #determine which genes get mutated
        noise = RNG.normal(loc=0.0, scale=MUTATION_STD, size=chrom.shape).astype(np.float32) #generate gaussian noise for each gene
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0) #apply noise only when gene is mutated
        mutated.append(new_chrom.astype(np.float32))
    return mutated


### fitness evaluation
def fitness_function(history: list[float]) -> float:
    """From template: calculate fitness as negative cartesian distance (--> maximization problem)"""
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def decode_and_build(genotype: List[np.ndarray]):
    """Decode NDE from a genotype and build robot (uses global NDE/HPD once per run).
       Returns (robot_graph, core) or None if unbuildable."""
    p_type, p_conn, p_rot = NDE.forward(genotype)
    robot_graph = HPD.probability_matrices_to_graph(p_type, p_conn, p_rot)
    try:
        core = construct_mjspec_from_graph(robot_graph)
    except Exception:
        # unbuildable body (e.g., invalid face for that module) → signal failure
        return None
    return robot_graph, core


def evaluate_genotype(genotype: List[np.ndarray]) -> Tuple[float, list]:
    """Run headless simulation (for now with nn_controller) and calculate fitness"""
    built = decode_and_build(genotype)
    if built is None:
        return -1e6, []  # prune unbuildable bodies

    robot_graph, core = built

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    if model.nu == 0:
        return -1e6, []
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    simple_runner(model, data, duration=SIM_DURATION, steps_per_loop=100)

    xpos_history = tracker.history.get("xpos", {})
    if not xpos_history or 0 not in xpos_history:
        return -1e6, []
    hist = xpos_history[0]
    return fitness_function(hist), hist


### EA loop
def run_ea():
    mj.set_mjcb_control(None)
    population = [random_genotype() for _ in range(POP_SIZE)]
    fitnesses, histories = [], []

    for i, gen in enumerate(population):
        fit, his = evaluate_genotype(gen)
        fitnesses.append(fit)
        histories.append(his)
        #FOR DEBUGGING:
        #print(f"startinng pop {i} fitness: {fit:.4f}")

    for gen_idx in range(GENS):
        print(f"\n=== Generation {gen_idx+1} ===")

        #calculate probs for roulette parent selection
        f_arr = np.array(fitnesses)
        weights = f_arr - f_arr.min() + 1e-6 #make all values positive and lowest fitness ~ 0
        probs = weights / weights.sum()

        #create pop size children and perform crossover and matation
        children = []
        while len(children) < POP_SIZE:
            pa, pb = RNG.choice(len(population), size=2, replace=False, p=probs)
            c1, c2 = crossover_per_chromosome(population[pa], population[pb])
            children.append(gaussian_mutation(c1))
            if len(children) < POP_SIZE:
                children.append(gaussian_mutation(c2))

        #calculate fitness for each child
        child_fitnesses, child_histories = [], []
        for i, chi in enumerate(children):
            fit, his = evaluate_genotype(chi)
            child_fitnesses.append(fit)
            child_histories.append(his)
            #FOR DEBUGGING:
            #print(f"Child {i} fitness: {fit:.4f}")

        #perform elitist survivor selection
        combined = population + children
        combined_fit = fitnesses + child_fitnesses
        combined_hist = histories + child_histories

        order = np.argsort(combined_fit)[::-1][:POP_SIZE]
        population = [combined[i] for i in order]
        fitnesses = [combined_fit[i] for i in order]
        histories = [combined_hist[i] for i in order]

        print("Survivors:")
        for r, f in enumerate(fitnesses):
            print(f"{r+1}: {f:.4f}")

    best_idx = int(np.argmax(fitnesses))
    best_gen, best_hist, best_fit = population[best_idx], histories[best_idx], fitnesses[best_idx]
    print("\n=== EA finished ===")
    print(f"Best fitness: {best_fit:.4f}")

    #save best robot json
    robot_graph, core = decode_and_build(best_gen)
    save_graph_as_json(robot_graph, OUTPUT / "best_robot.json")
    print(f"Saved best robot graph to {OUTPUT/'best_robot.json'}")

    #save best robot video
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    video_folder = OUTPUT / "videos"
    video_folder.mkdir(exist_ok=True)
    recorder = VideoRecorder(output_folder=str(video_folder))
    video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    print(f"Saved video of best robot to {video_folder}")

    return best_gen, best_hist, best_fit


if __name__ == "__main__":
    run_ea()
