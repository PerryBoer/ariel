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
import csv
from datetime import datetime

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
MIN_VIABLE_MOVEMENT = 0.015

# data directory
CWD = Path.cwd()
OUTPUT = CWD / "__output__"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Timestamp for output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set NDE (constant for entire run)
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)

# Controller functions
def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    """From template: random nn controller, used to check robot viability"""
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

def cpg_controller(
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

def decode_and_build(NDE, genotype: List[np.ndarray]):
    """Decode nde from a genotype and build robot"""
    p_type, p_conn, p_rot = NDE.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core


def evaluate_genotype_old(genotype: List[np.ndarray]) -> Tuple[float, list]:
    """Run headless simulation and calculate fitness"""
    robot_graph, core = decode_and_build(NDE,genotype)

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=cpg_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    simple_runner(model, data, duration=SIM_DURATION, steps_per_loop=100)

    xpos_history = tracker.history.get("xpos", {})
    if len(xpos_history) == 0 or 0 not in xpos_history:
        return -1e6, []
    hist = xpos_history[0]
    return fitness_function(hist), hist

def check_viability(genotype: List[np.ndarray], MIN_VIABLE_MOVEMENT:float):
    """Run a 6 sec random simulation and check if the robot has moved > threshold"""
    robot_graph, core = decode_and_build(NDE,genotype)

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker) #run with random nn controller to check movement
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    simple_runner(model, data, duration=6, steps_per_loop=100)
    xpos_history = tracker.history.get("xpos", {})
    hist = xpos_history[0]

    pos_3 = hist[3] #xyz position of body at 3 seconds (to ignore initial 'wobble')
    pos_final = hist[-1] #xyz position of body at end of 6 sec simulation
    pos_diff = pos_3 - pos_final
    if pos_diff[0] > MIN_VIABLE_MOVEMENT or pos_diff[1] > MIN_VIABLE_MOVEMENT: #check movement in either x or y direction
        viability = True
    else: 
        viability = False

    return viability, hist

def evaluate_genotype(genotype: List[np.ndarray], MIN_VIABLE_MOVEMENT:float):
    viability, viability_hist = check_viability(genotype, MIN_VIABLE_MOVEMENT)
    if viability == True: 
        #train CPG
        return fitness_function(viability_hist), viability_hist #CHANGE
    else:
        return fitness_function(viability_hist), viability_hist


### EA loop
def run_ea():
    mj.set_mjcb_control(None)
    population = [random_genotype() for _ in range(POP_SIZE)]
    fitnesses, histories, best_fit_per_gen = [], [], []

    for i, gen in enumerate(population):
        fit, his = evaluate_genotype(gen, MIN_VIABLE_MOVEMENT)
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
            fit, his = evaluate_genotype(chi, MIN_VIABLE_MOVEMENT)
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

        print(f"Best fitness: {fitnesses[0]:.4f}")
        best_fit_per_gen.append(fitnesses[0])

    best_idx = int(np.argmax(fitnesses))
    best_gen, best_hist, best_fit = population[best_idx], histories[best_idx], fitnesses[best_idx]
    print("\n=== EA finished ===")
    print(f"Best fitness: {best_fit:.4f}")
   

    # Save best robot graph
    robot_graph, core = decode_and_build(NDE,best_gen)
    graph_folder = OUTPUT / "best_robot_graphs"
    graph_folder.mkdir(exist_ok=True)
    graph_file = f"best_robot_{TIMESTAMP}.json"
    save_graph_as_json(robot_graph, graph_folder / graph_file)
    print(f"\nSaved best robot graph to {graph_folder / graph_file}")

    # Save best fitness per generation
    fitness_folder = OUTPUT / "fitness_per_gen"
    fitness_folder.mkdir(exist_ok=True)
    fitness_file = f"fitness_{TIMESTAMP}.csv"
    with open(fitness_folder/fitness_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(best_fit_per_gen)
    print(f"Saved best fitness values per generation to {fitness_folder/fitness_file}")

    # Save video of best robot (use longer duration for clarity)
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(controller_callback_function=cpg_controller, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    video_folder = OUTPUT / "videos"
    video_folder.mkdir(exist_ok=True)
    recorder = VideoRecorder(output_folder=str(video_folder))
    video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    print(f"Saved video of best robot to {video_folder}")

    return best_gen, best_hist, best_fit


if __name__ == "__main__":
    run_ea()
