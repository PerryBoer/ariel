"""Assignment 3 – EA over bodies + RevDE-optimized CPG controller (minimal, robust)."""

from pathlib import Path
from typing import Any, List, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime
import csv

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libs
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
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

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# ---------------- Global params ----------------
SEED = 42
RNG = np.random.default_rng(SEED)

# EA (bodies)
POP_SIZE = 8
GENS = 5
GENE_LEN = 64
MUTATION_STD = 0.08
MUTATION_PROB = 0.1
CROSSOVER_PROB = 0.8

# Inner loop / CPG RevDE
REVDE_POP = 18
REVDE_GENS = 16
DE_F = 0.6
DE_CR = 0.7

# Sim / world
SPAWN_POS = [-0.8, 0, 0.20]               # a bit higher than default
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
SIM_DURATION = 10.0

# Viability: very relaxed (just not completely stuck)
MIN_VIABLE_XY_DISP = 0.01                 # meters between ~3s and end

# Paths
SCRIPT_NAME = "A3_revde_baseline"
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Fixed NDE for whole run (as per assignment)
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)


# ---------------- Fitness & helpers ----------------
def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)


def decode_to_graph(genotype: List[np.ndarray]):
    p_type, p_conn, p_rot = NDE.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(p_type, p_conn, p_rot)
    return graph


def preflight_body(graph, min_nodes=3) -> Tuple[bool, int]:
    """Lightweight check: can compile & spawn? has actuators? not trivially tiny?"""
    try:
        core = construct_mjspec_from_graph(graph)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS,
                    correct_for_bounding_box=True, small_gap=0.01)
        model = world.spec.compile()
        nu = int(model.nu)
        if nu <= 0:
            return False, 0
        if len(graph.nodes) < min_nodes:
            return False, nu
        return True, nu
    except Exception as e:
        console.log(f"[preflight] rejected (spawn/compile): {e}")
        return False, 0


# ---------------- Controllers ----------------
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
    """Random 3-layer NN (only for quick viability runs)."""
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu
    w1 = RNG.normal(0.0138, 0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(0.0138, 0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(0.0138, 0.5, size=(hidden_size, output_size))
    x = data.qpos
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    u = np.tanh(h2 @ w3) * np.pi
    return u


class AdvancedCPG:
    """
    Per-actuator sinusoid:
        u_i(t) = clip( A_i * sin(omega * t + phi_i) + b_i, [-pi/2, +pi/2] )
    Packed params x = [A(0..nu-1), phi(0..nu-1), b(0..nu-1), omega]
    Bounds:
        0 <= A_i <= 0.6*pi/2
        -pi <= phi_i <= pi
        -0.25*pi <= b_i <= 0.25*pi
        0.2 <= omega <= 2*pi   (~0.03Hz..1Hz)
    """
    def __init__(self, nu: int, x: np.ndarray):
        self.nu = int(nu)
        self.x = x.astype(np.float64, copy=False)
        self._unpack()

    @staticmethod
    def bounds(nu: int) -> Tuple[np.ndarray, np.ndarray]:
        CONTROL_BOUND = np.pi / 2
        Amax = 0.6 * CONTROL_BOUND
        lo = np.concatenate([
            np.zeros(nu),
            -np.pi * np.ones(nu),
            -0.25*np.pi * np.ones(nu),
            np.array([0.2]),
        ])
        hi = np.concatenate([
            Amax * np.ones(nu),
            +np.pi * np.ones(nu),
            +0.25*np.pi * np.ones(nu),
            np.array([2.0*np.pi]),
        ])
        return lo, hi

    @staticmethod
    def random(nu: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = AdvancedCPG.bounds(nu)
        return rng.uniform(lo, hi).astype(np.float64)

    def _unpack(self):
        nu = self.nu
        self.A = self.x[0:nu]
        self.phi = self.x[nu:2*nu]
        self.bias = self.x[2*nu:3*nu]
        self.omega = float(self.x[3*nu])

    def action(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        t = float(data.time)
        u = self.A * np.sin(self.omega * t + self.phi) + self.bias
        np.clip(u, -np.pi/2, np.pi/2, out=u)
        return u


# ---------------- Genotype ops (bodies) ----------------
def random_genotype() -> List[np.ndarray]:
    # softer init than uniform: Beta(2,2) reduces extreme logits in HPD
    def beta_vec():
        return RNG.beta(2.0, 2.0, size=GENE_LEN).astype(np.float32)
    return [beta_vec(), beta_vec(), beta_vec()]


def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = a.size
    p = int(RNG.integers(1, L))
    c1 = np.concatenate([a[:p], b[p:]])
    c2 = np.concatenate([b[:p], a[p:]])
    return c1.astype(np.float32), c2.astype(np.float32)


def crossover_per_chromosome(pa: List[np.ndarray], pb: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    c1, c2 = [], []
    for idx in range(3):
        if RNG.random() < CROSSOVER_PROB:
            a, b = one_point_crossover(pa[idx], pb[idx])
            c1.append(a); c2.append(b)
        else:
            c1.append(pa[idx].copy()); c2.append(pb[idx].copy())
    return c1, c2


def gaussian_mutation(gen: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for chrom in gen:
        mask = RNG.random(chrom.shape) < MUTATION_PROB
        noise = RNG.normal(0.0, MUTATION_STD, size=chrom.shape).astype(np.float32)
        x = np.clip(chrom + noise * mask, 0.0, 1.0)
        out.append(x.astype(np.float32))
    return out


# ---------------- Viability (very relaxed) ----------------
def quick_viability(graph) -> Tuple[bool, list]:
    """6s random NN; viable if ||p(end)-p(3s)||_2 in XY >= tiny threshold."""
    ok, _ = preflight_body(graph)
    if not ok:
        return False, []

    core = construct_mjspec_from_graph(graph)
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS,
                correct_for_bounding_box=True, small_gap=0.01)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    tracker.setup(world.spec, data)

    ctrl = Controller(nn_controller, tracker=tracker, alpha=0.5, time_steps_per_ctrl_step=150)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    simple_runner(model, data, duration=6.0, steps_per_loop=100)

    hist = tracker.history.get("xpos", {}).get(0, [])
    if len(hist) < 3:
        return False, hist
    # map ~3s index robustly
    i3 = max(0, min(int(round((3.0/6.0)*(len(hist)-1))), len(hist)-1))
    p3 = np.array(hist[i3]); pend = np.array(hist[-1])
    if np.linalg.norm((pend - p3)[:2]) >= MIN_VIABLE_XY_DISP:
        return True, hist
    return False, hist


# ---------------- RevDE inner loop ----------------
def optimize_cpg_revde(graph, duration: float, rng: np.random.Generator):
    ok, nu = preflight_body(graph)
    if not ok:
        return None, -1e6, []

    lo, hi = AdvancedCPG.bounds(nu)
    dim = 3*nu + 1

    def eval_params(x: np.ndarray) -> Tuple[float, list]:
        try:
            core = construct_mjspec_from_graph(graph)
            mj.set_mjcb_control(None)
            world = OlympicArena()
            world.spawn(core.spec, spawn_position=SPAWN_POS,
                        correct_for_bounding_box=True, small_gap=0.01)
            model = world.spec.compile()
            data = mj.MjData(model)
            mj.mj_resetData(model, data)

            tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
            tracker.setup(world.spec, data)

            cpg = AdvancedCPG(nu, x)
            ctrl = Controller(lambda m, d: cpg.action(m, d), tracker=tracker, alpha=0.9)
            mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
            simple_runner(model, data, duration=duration, steps_per_loop=100)

            hist = tracker.history.get("xpos", {}).get(0, [])
            if not hist:
                return -1e6, []
            return fitness_function(hist), hist
        except Exception as e:
            console.log(f"[RevDE eval] spawn/compile error: {e}")
            return -1e6, []

    # init
    pop = np.stack([AdvancedCPG.random(nu, rng) for _ in range(REVDE_POP)], axis=0)
    fits = np.empty(REVDE_POP, dtype=float)
    hists: List[Any] = [None] * REVDE_POP
    for i in range(REVDE_POP):
        fits[i], hists[i] = eval_params(pop[i])

    # loop
    for g in range(REVDE_GENS):
        order = rng.permutation(REVDE_POP)
        for k in range(0, REVDE_POP, 2):
            i = order[k]
            j = order[(k+1) % REVDE_POP]
            donors = [d for d in range(REVDE_POP) if d not in (i, j)]
            if len(donors) < 3:
                continue
            a, b, c = rng.choice(donors, size=3, replace=False)

            # i side
            vi = pop[a] + DE_F * (pop[b] - pop[c])
            vi = np.clip(vi, lo, hi)
            mask = rng.random(dim) < DE_CR
            if not mask.any():
                mask[rng.integers(0, dim)] = True
            ui = np.where(mask, vi, pop[i])
            fi, hi_ = eval_params(ui)
            if fi > fits[i]:
                pop[i], fits[i], hists[i] = ui, fi, hi_

            # j side (reversible)
            vj = pop[a] + DE_F * (pop[c] - pop[b])
            vj = np.clip(vj, lo, hi)
            mask2 = rng.random(dim) < DE_CR
            if not mask2.any():
                mask2[rng.integers(0, dim)] = True
            uj = np.where(mask2, vj, pop[j])
            fj, hj_ = eval_params(uj)
            if fj > fits[j]:
                pop[j], fits[j], hists[j] = uj, fj, hj_
        console.log(f"[RevDE] gen {g+1}/{REVDE_GENS} | best={np.max(fits):.4f} | mean={np.mean(fits):.4f}")

    best = int(np.argmax(fits))
    return pop[best], float(fits[best]), hists[best]


# ---------------- Full evaluation of a body ----------------
def evaluate_genotype(genotype: List[np.ndarray]) -> Tuple[float, list, np.ndarray | None]:
    graph = decode_to_graph(genotype)

    viable, vhist = quick_viability(graph)
    if not viable:
        return -1e6, vhist, None

    best_cpg, best_fit, best_hist = optimize_cpg_revde(graph, SIM_DURATION, RNG)
    return best_fit, best_hist, best_cpg


# ---------------- EA over bodies ----------------
def run_ea():
    mj.set_mjcb_control(None)

    population = [random_genotype() for _ in range(POP_SIZE)]
    fitnesses: List[float] = []
    histories: List[list] = []
    cpg_params: List[Any] = []

    # initial evals
    for i, g in enumerate(population):
        f, h, c = evaluate_genotype(g)
        fitnesses.append(f); histories.append(h); cpg_params.append(c)
        console.log(f"init {i}: fitness={f:.4f}")

    best_fit_per_gen: List[float] = []

    for gen_idx in range(GENS):
        print(f"\n=== Generation {gen_idx+1} ===")
        f_arr = np.array(fitnesses, dtype=float)
        weights = f_arr - f_arr.min() + 1e-6
        probs = weights / weights.sum()

        # offspring
        children: List[List[np.ndarray]] = []
        while len(children) < POP_SIZE:
            pa, pb = RNG.choice(len(population), size=2, replace=False, p=probs)
            c1, c2 = crossover_per_chromosome(population[pa], population[pb])
            children.append(gaussian_mutation(c1))
            if len(children) < POP_SIZE:
                children.append(gaussian_mutation(c2))

        child_fits: List[float] = []
        child_hists: List[list] = []
        child_cpgs: List[Any] = []
        for i, chi in enumerate(children):
            f, h, c = evaluate_genotype(chi)
            child_fits.append(f); child_hists.append(h); child_cpgs.append(c)
            console.log(f"child {i}: fitness={f:.4f}")

        # elitist survivor selection
        combined = population + children
        combined_fit = fitnesses + child_fits
        combined_hist = histories + child_hists
        combined_cpg = cpg_params + child_cpgs
        order = np.argsort(combined_fit)[::-1][:POP_SIZE]
        population = [combined[i] for i in order]
        fitnesses = [combined_fit[i] for i in order]
        histories = [combined_hist[i] for i in order]
        cpg_params = [combined_cpg[i] for i in order]

        print(f"Best fitness: {fitnesses[0]:.4f}")
        best_fit_per_gen.append(fitnesses[0])

    # finalize
    best_idx = int(np.argmax(fitnesses))
    best_gen, best_hist, best_fit, best_cpg = population[best_idx], histories[best_idx], fitnesses[best_idx], cpg_params[best_idx]
    print("\n=== EA finished ==="); print(f"Best fitness: {best_fit:.4f}")

    # save graph
    best_graph = decode_to_graph(best_gen)
    save_graph_as_json(best_graph, DATA / f"best_robot_{TIMESTAMP}.json")
    print(f"Saved best robot graph to {DATA / f'best_robot_{TIMESTAMP}.json'}")

    # save fitness curve
    with (DATA / f"fitness_{TIMESTAMP}.csv").open("w", newline="") as f:
        csv.writer(f).writerow(best_fit_per_gen)
    print(f"Saved best fitness-per-gen CSV to {DATA / f'fitness_{TIMESTAMP}.csv'}")

    # save video of best
    try:
        core = construct_mjspec_from_graph(best_graph)
        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(core.spec, spawn_position=SPAWN_POS,
                    correct_for_bounding_box=True, small_gap=0.01)
        model = world.spec.compile()
        data = mj.MjData(model); mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core"); tracker.setup(world.spec, data)
        if best_cpg is None:
            # tiny fallback: global sine
            def fixed_cpg(m, d):
                nu = m.nu; t = float(d.time); A = 0.6*(np.pi/2)
                ph = 2*np.pi*(np.arange(nu)/max(1, nu))
                u = A*np.sin(2*np.pi*0.7*t + ph); np.clip(u, -np.pi/2, np.pi/2, out=u)
                return u
            ctrl = Controller(fixed_cpg, tracker=tracker)
        else:
            cpg = AdvancedCPG(int(model.nu), best_cpg)
            ctrl = Controller(lambda m, d: cpg.action(m, d), tracker=tracker, alpha=0.9)

        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
        video_folder = DATA / "videos"; video_folder.mkdir(exist_ok=True)
        rec = VideoRecorder(output_folder=str(video_folder))
        video_renderer(model, data, duration=SIM_DURATION, video_recorder=rec)
        print(f"Saved video to {video_folder}")
    except Exception as e:
        print(f"[video save] error: {e}")

    return best_gen, best_hist, best_fit


# ---------------- If you want the single-robot demo from template ----------------
def show_xpos_history(history: list[float]) -> None:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]; camera.distance = 10; camera.azimuth = 0; camera.elevation = -90
    mj.set_mjcb_control(None)
    world = OlympicArena(); model = world.spec.compile(); data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, camera=camera, save_path=save_path, save=True)
    img = plt.imread(save_path); _, ax = plt.subplots(); ax.imshow(img)
    w, h, _ = img.shape
    pos = np.array(history)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pts = [[xc, yc]]
    for i in range(len(pos) - 1):
        xd, yd, _ = pos[i+1] - pos[i]
        xn, yn = pts[i]
        pts.append([xn + int(xd / pixel_to_dist), yn + int(yd / pixel_to_dist)])
    pts = np.array(pts)
    ax.plot(x0, y0, "kx", label="[0, 0, 0]"); ax.plot(xc, yc, "go", label="Start")
    ax.plot(pts[:, 0], pts[:, 1], "b-", label="Path"); ax.plot(pts[-1, 0], pts[-1, 1], "ro", label="End")
    ax.set_xlabel("X Position"); ax.set_ylabel("Y Position"); ax.legend(); plt.title("Robot Path in XY Plane"); plt.show()


# ---------------- Run ----------------
if __name__ == "__main__":
    run_ea()
