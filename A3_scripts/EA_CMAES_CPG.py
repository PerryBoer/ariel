"""Assignment 3 – Minimal GA + per-body CMA-ES CPG + export of best robot video and JSON."""

# ---------- Imports ----------
from pathlib import Path
from typing import TYPE_CHECKING, Any
import math
import numpy as np
import numpy.typing as npt
import mujoco as mj
import csv
from datetime import datetime

import torch
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

# Outer EA loop parameters (evolving the body)
POP_SIZE = 10
N_GEN = 100
CX_PROB = 0.6
MUT_PROB = 0.5 
MUT_SIGMA = 0.2
ELITISM_SIZE = 2 # Best n always kept for the next generation
PICK_PARENTS_BETA = 5 #higher value -> favors higher fitnesses to be picked as parents
if N_GEN > 30:
    SIM_TIME_STAGES = [10.0, 20.0, 40.0, 70.0] #dynamic simulation time based on gen num (per 25% of total generations)
else: SIM_TIME_STAGES = [10.0, 10.0, 15.0, 20.0] #for test-running short runs

#Inner EA loop parameters (evolving the CPG/brain)
MIN_VIABLE_MOVEMENT = 0.015 # Mimimal movement in a random 6s simulation to be viable
CPG_TRAINING_POP = 10
CPG_TRAINING_GENS = 15 
PHASE_MIN, PHASE_MAX = -math.pi, math.pi
AMP_MIN, AMP_MAX     = 0.0, 1.0
FREQ_MIN, FREQ_MAX   = 0.4, 2.0
SMOOTH_ALPHA         = 0.5

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

# ---------- Random controller (kept for body viability check) ----------
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    input_size = len(data.qpos)
    hidden = 8
    output = model.nu
    if output == 0:
        return np.zeros(0)
    w1 = RNG.normal(0.0, 0.5, size=(input_size, hidden))
    w2 = RNG.normal(0.0, 0.5, size=(hidden, output))
    return np.tanh(np.tanh(data.qpos @ w1) @ w2) * np.pi

# ---------- Fitness (from template) ----------
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
        to_mut = RNG.random(len(chrom)) < mut_prob #determine which genes get mutated
        noise = RNG.normal(loc=0.0, scale=sigma, size=len(chrom)).astype(np.float32) #generate gaussian noise for each gene
        new_chrom = np.clip(chrom + noise * to_mut, 0.0, 1.0) #apply noise only when gene is mutated
        mutated.append(new_chrom.astype(np.float32))
    return mutated

# ---------- CPG training ----------
def _find_core_geom_id(model: mj.MjModel) -> int | None:
    # Best-effort: find a geom with 'core' in its name for tracking last xyz
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name.lower():
            return gid
    return None

def make_cpg_controller(params: np.ndarray, model: mj.MjModel, alpha: float = SMOOTH_ALPHA):
    """params = [phase_0..phase_{nu-1}, AMP, FREQ]"""
    nu = int(model.nu)
    phases = np.asarray(params[:nu], dtype=np.float64) if nu > 0 else np.zeros(0, dtype=np.float64)
    AMP    = float(params[nu]) if nu > 0 else 0.0
    FREQ   = float(params[nu + 1]) if nu > 0 else 1.0

    # Safety clamps
    phases = np.clip(phases, PHASE_MIN, PHASE_MAX)
    AMP  = float(np.clip(AMP, AMP_MIN, AMP_MAX))
    FREQ = float(np.clip(FREQ, FREQ_MIN, FREQ_MAX))

    lo = model.actuator_ctrlrange[:, 0] if nu > 0 else np.zeros(0)
    hi = model.actuator_ctrlrange[:, 1] if nu > 0 else np.zeros(0)
    center    = 0.5 * (hi + lo)
    half_span = 0.5 * (hi - lo)

    def cb(m: mj.MjModel, d: mj.MjData):
        if nu == 0:
            return
        t = d.time
        y = AMP * np.sin(2.0 * math.pi * FREQ * t + phases)    # [-AMP, AMP]
        y = np.clip(y, -1.0, 1.0)
        target = center + half_span * y
        np.clip(target, lo, hi, out=target)
        d.ctrl[:] = d.ctrl + alpha * (target - d.ctrl)

    return cb

class BodyCPGProblem(Problem):
    """EvoTorch problem wrapping a *given compiled model* for a decoded body."""
    def __init__(self, model: mj.MjModel, sim_seconds: float = 6.0):
        self.model = model
        self.sim_seconds = float(sim_seconds)
        self.steps_per_sec = int(round(1.0 / model.opt.timestep))
        self.core_gid = _find_core_geom_id(model)

        nu = int(model.nu)
        L = (nu + 2) if nu > 0 else 2  # still define AMP,FREQ even if nu==0 (won't be used)

        lo = np.concatenate([np.full(nu, PHASE_MIN), [AMP_MIN], [FREQ_MIN]]).astype(np.float64)
        hi = np.concatenate([np.full(nu, PHASE_MAX), [AMP_MAX], [FREQ_MAX]]).astype(np.float64)

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

        # Controller
        cb = make_cpg_controller(theta, self.model)
        mj.set_mjcb_control(cb)

        horizon = int(round(self.sim_seconds * self.steps_per_sec))
        # default: fallback to body position if core geom not found
        xyz_last = None
        try:
            for _ in range(horizon):
                mj.mj_step(self.model, data)
                if self.core_gid is not None:
                    xyz_last = data.geom_xpos[self.core_gid].copy()
                else:
                    # body root (qpos first 3) as fallback
                    xyz_last = np.array([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2] if self.model.nq >= 3 else 0.0)])
        finally:
            mj.set_mjcb_control(None)

        if xyz_last is None:
            return 1e6  # failed sim -> terrible distance
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
    """Run a short CMA-ES on the CPG params for THIS body; return best params (numpy)."""
    nu = int(model.nu)
    # Degenerate case: no actuators -> return sensible dummy
    if nu == 0:
        return np.array([0.5, 1.0], dtype=np.float64)  # AMP, FREQ (unused)

    prob = BodyCPGProblem(model, sim_seconds=seconds)
    center = np.concatenate([np.zeros(nu), [0.5], [1.0]]).astype(np.float64)
    solver = CMAES(
        prob,
        popsize=max(10, int(pop)),
        stdev_init=0.3,
        center_init=torch.from_numpy(center),
    )
    _ = StdOutLogger(solver, interval=max(1, gens // 5))

    best_theta = center.copy()
    best_eval = float("inf")

    for _ in range(int(gens)):
        solver.step()
        pop = solver.population
        # Use the batch’s values and evals
        vals = pop.values.detach().cpu().numpy()   
        fits_t = pop.evals                         
        fits = fits_t.detach().cpu().numpy().reshape(-1)
        i = int(np.argmin(fits))
        if fits[i] < best_eval:
            best_eval = float(fits[i])
            best_theta = vals[i].copy()

    return best_theta

# ---------- Check viability (able to move)----------
def decode_and_build(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray]):
    """Decode nde from a genotype and build robot"""
    p_type, p_conn, p_rot = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)
    return robot_graph, core

def check_viability(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray], min_viable_movement:float):
    """Run a 6 sec random simulation and check if the robot has moved > threshold"""
    _, core = decode_and_build(nde,genotype)

    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
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
    viability = (abs(pos_diff[0]) > min_viable_movement or abs(pos_diff[1]) > min_viable_movement)

    return viability, hist

# ---------- Evaluation (train CPG per body and report best fitness) ----------
def evaluate(nde: NeuralDevelopmentalEncoding, genotype: list[np.ndarray], sim_time: float) -> tuple[float, "DiGraph", np.ndarray]:
    """Check viability -> if viable train CPG via CMA-ES -> simulate with tracker -> fitness."""
    # Check viability
    viable, hist = check_viability(nde, genotype, MIN_VIABLE_MOVEMENT)
    if not viable:
        # Return bad fitness and empty graph/theta for non-viable bodies
        console.log("[bold red]Body was not viable, skipping CPG training[/bold red]")
        return -10, None, None #return very low fitness for non viable bodies
    
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    # Compile model for THIS body
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, position=SPAWN_POS)
    model = world.spec.compile()

    # Inner-loop: train a CPG for this specific body
    theta = optimize_cpg_cma_for_body(model, seconds=sim_time, pop=CPG_TRAINING_POP, gens=CPG_TRAINING_GENS)

    # Now re-run a normal tracked experiment (your existing pipeline) with best CPG
    def cpg_callback(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        # we piggyback on Controller which calls this each step; return d.ctrl after updating
        cb = make_cpg_controller(theta, m)
        cb(m, d)
        return d.ctrl

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=cpg_callback, tracker=tracker)

    # Compile new body from same genotype to prevent compilation error
    p_type, p_conn, p_rot = nde.forward(genotype)
    robot_graph: "DiGraph" = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    core = construct_mjspec_from_graph(robot_graph)

    experiment(robot=core, controller=ctrl, duration=sim_time)

    # Use original fitness on the tracked history
    hist = tracker.history["xpos"][0]
    fit = fitness_function(hist)
    return fit, robot_graph, theta

# ---------- Initialize viable population ----------
def initialize_viable_population(nde: NeuralDevelopmentalEncoding, pop_size: int)-> list[np.ndarray]:
    population = []

    while len(population) < pop_size:
        geno = random_genotype()
        viable, _ = check_viability(nde, geno, MIN_VIABLE_MOVEMENT)
        if viable:
            population.append(geno)

    return population

# ---------- Probabilistic parent selection ----------
def pick_parents(population: list[np.ndarray], fitnesses: list[float], beta: float):
    """Pick one parent from population with probability exp(beta * fitness)."""
    shifted = fitnesses - fitnesses.min() #shift so lowest fitness is 0
    probs = np.exp(beta * shifted)
    probs /= probs.sum()
    idx1 = RNG.choice(len(population), p=probs)
    # try up to 10 times to pick a different parent
    for attempt in range(10):
        idx2 = RNG.choice(len(population), p=probs)
        if idx2 != idx1:
            break
    else:
        idx2 = idx1
        console.log(f"[red]Warning: Could not pick distinct parent after 10 attempts, using same parent twice[/red]")

    return population[idx1], population[idx2]

# ---------- EA main ----------
def main() -> None:
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES) # Set NDE once, constant for entire run
    console.log(f"[bold cyan]Starting EA with CMA-ES CPG for {N_GEN} generations, pop={POP_SIZE}[/bold cyan]")

    population = initialize_viable_population(nde, POP_SIZE)
    best_fit = -np.inf
    best_graph = None
    best_theta = None

    # Setup best fitness output csv file (appended at each generation)
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
            console.log(f"Robot {i:02d} -> fitness = {fit:.4f} (sim time {sim_time:.1f}s)")
            if fit > best_fit:
                best_fit, best_graph, best_theta = fit, graph, theta

        # Parent selection (elitism and probabilistic)
        fitnesses = np.maximum(fitnesses, -10.0) #failsave to set all fitnesses to >-10, to prevent overflow error in pick_parents() when a robot falls of the platform
        sorted_idx = np.argsort(fitnesses)
        elite_idx = sorted_idx[-ELITISM_SIZE:] #top-n elites are always kept (unchanged) for the next generation
        elites = [population[i] for i in elite_idx]
        console.log(f"Best fitness={fitnesses[elite_idx[-1]]:.4f}")
        new_pop = elites.copy()
        while len(new_pop) < POP_SIZE:
            #p1, p2 = RNG.choice(parent_pool, 2, replace=True)
            p1, p2 = pick_parents(population, fitnesses, beta=PICK_PARENTS_BETA)
            child = crossover_per_chromosome(p1, p2, CX_PROB)[0] #only use one of the two crossover children to increase exploration
            child = gaussian_mutation(child, MUT_PROB, MUT_SIGMA)
            new_pop.append(child)
        population = new_pop

        # Save best fitness per gen in csv
        with open(fitness_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fit])
        
        # Save best video, body json and CPG params at 10 checkpoints
        if N_GEN >9 and (gen + 1) % (N_GEN // 10) == 0:
            console.log(f"[yellow]Checkpoint: saving video, json and paramaters at generation {gen}[/yellow]")
            best_core = construct_mjspec_from_graph(best_graph)

            def cpg_video_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
                cb = make_cpg_controller(best_theta, m)
                cb(m, d)
                return d.ctrl

            tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
            ctrl = Controller(controller_callback_function=cpg_video_controller, tracker=tracker)

            video_folder = DATA / "videos"
            video_folder.mkdir(exist_ok=True)
            experiment(robot=best_core, controller=ctrl, duration=sim_time, record=True)
            console.log(f"[green]Saved checkpoint video in {video_folder}[/green]")

            #save current best body json
            graph_folder = DATA / "best_robot_graphs_checkpoints"
            graph_folder.mkdir(exist_ok=True)
            graph_file = f"best_robot_{TIMESTAMP}_gen{gen}.json"
            save_graph_as_json(best_graph, graph_folder / graph_file)
            console.log(f"[green]Saved checkpoint body json at {graph_folder / graph_file}[/green]")

            #save current best CPG params
            params_folder = DATA / "best_robot_params_checkpoints"
            params_folder.mkdir(exist_ok=True)
            params_file = f"best_robot_{TIMESTAMP}_gen{gen}.txt"
            with open(params_folder/params_file, "w") as f:
                f.write(", ".join(map(str,best_theta)))
            console.log(f"[green]Saved checkpoint CPG params at {params_folder/params_file}[/green]")

    # ---------- After final generation ----------
    console.rule("[bold magenta]Final best robot[/bold magenta]")
    console.log(f"Best fitness = {best_fit:.4f}")

    # Save body graph json for best robot
    graph_folder = DATA / "best_robot_graphs"
    graph_folder.mkdir(exist_ok=True)
    graph_file = f"best_robot_{TIMESTAMP}.json"
    save_graph_as_json(best_graph, graph_folder / graph_file)
    console.log(f"\n[green]Saved best robot graph to {graph_folder / graph_file}[/green]")

    # Save CPG parameters for best robot
    params_folder = DATA / "best_robot_params"
    params_folder.mkdir(exist_ok=True)
    params_file = f"best_robot_{TIMESTAMP}.txt"
    with open(params_folder/params_file, "w") as f:
        f.write(", ".join(map(str,best_theta)))
    console.log(f"\n[green]Saved best robot parameters to {params_folder/params_file}[/green]")

    # Save video for best robot with trained CPG
    best_core = construct_mjspec_from_graph(best_graph)

    def cpg_video_controller(m: mj.MjModel, d: mj.MjData) -> npt.NDArray[np.float64]:
        cb = make_cpg_controller(best_theta, m)
        cb(m, d)
        return d.ctrl

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=cpg_video_controller, tracker=tracker)
    console.log("[yellow]Recording video of best robot...[/yellow]")
    video_folder = DATA / "videos"
    video_folder.mkdir(exist_ok=True)
    experiment(robot=best_core, controller=ctrl, duration=sim_time, record=True)
    console.log(f"[green]All done! Video and graph saved.[/green], at {video_folder}")
    

if __name__ == "__main__":
    main()
