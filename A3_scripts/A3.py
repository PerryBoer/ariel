"""A3 — Body-encoding EA (NDE only, fixed nn_controller)

- Keep nn_controller() and experiment() exactly as in the template.
- One global NDE + HPD (deterministic decoding).
- Population = 20. (1+19) EA for a few generations.
- One-point crossover (shared cut across 3 input vectors).
- Gaussian mutation (gentle: sigma=0.05, p=0.10), clipped to [0,1].
- Graph "repair": remove 'NONE', keep core component, BFS tree, one child per face.
- Log *everything* (including repair stats) to CSV.
- Save raw & sanitized graphs; record video for elite each generation.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Tuple, List, Dict, Set
import copy
import csv
import time

import numpy as np
import numpy.typing as npt
import mujoco as mj
import matplotlib.pyplot as plt
from mujoco import viewer
import networkx as nx

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

if TYPE_CHECKING:
    from networkx import DiGraph

type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# === Randomness / Paths / Globals ==================================================
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
VID_DIR = DATA / "videos"
(DATA / "graphs_raw").mkdir(parents=True, exist_ok=True)
(DATA / "graphs_sanitized").mkdir(parents=True, exist_ok=True)
VID_DIR.mkdir(parents=True, exist_ok=True)

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
CORE_NODE = 0

# EA config
POP_SIZE = 20
GENERATIONS = 5
VIDEO_ELITE_ONLY = True          # set False to record video for everyone (expensive!)
SIM_DURATION_S = 6               # short clips to keep runs brisk

# One global NDE + HPD for determinism
GLOBAL_NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
GLOBAL_HPD = HighProbabilityDecoder(NUM_OF_MODULES)

# === Helpers (kept as in template) ================================================
def fitness_function(history: list[float]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    dist = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -dist


def show_xpos_history(history: list[float]) -> None:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(model, data, camera=camera, save_path=save_path, save=True)

    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    pos_data = np.array(history)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.show()


# === KEEP nn_controller EXACTLY AS TEMPLATE =======================================
def nn_controller(model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
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


# === KEEP experiment EXACTLY AS TEMPLATE ==========================================
def experiment(
    robot: Any, controller: Controller, duration: int = 15, mode: ViewerTypes = "viewer"
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    args: list[Any] = []
    kwargs: dict[Any, Any] = {}

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, *args, **kwargs))

    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "frame":
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            path_to_video_folder = str(VID_DIR)
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        case "launcher":
            viewer.launch(model=model, data=data)
        case "no_control":
            mj.set_mjcb_control(None)
            viewer.launch(model=model, data=data)
    # ==================================================================== #


# === Genotype utils ==============================================================

def random_genotype(geno_len: int = 64) -> List[np.ndarray]:
    """Return [type, connect, rotate] each ~ Beta(2,2) in [0,1]^geno_len as float32."""
    def beta_vec():
        return RNG.beta(2.0, 2.0, size=geno_len).astype(np.float32)
    return [beta_vec(), beta_vec(), beta_vec()]


def one_point_crossover(
    g1: List[np.ndarray], g2: List[np.ndarray], rng: np.random.Generator
) -> List[np.ndarray]:
    """Per-vector one-point crossover; same cut for the 3 inputs to preserve coordination."""
    L = g1[0].shape[0]
    cut = int(rng.integers(1, L))  # in [1, L-1]
    return [np.concatenate([g1[k][:cut], g2[k][cut:]]).astype(np.float32) for k in range(3)]


def gaussian_mutation(
    g: List[np.ndarray],
    rng: np.random.Generator,
    sigma: float = 0.05,
    p: float = 0.10,
) -> List[np.ndarray]:
    """Add N(0, sigma^2) to a Bernoulli(p)-masked subset of genes; clip to [0,1]."""
    mutated = []
    for vec in g:
        v = vec.copy()
        mask = rng.random(v.shape[0]) < p
        noise = rng.normal(0.0, sigma, size=v.shape[0]).astype(np.float32)
        v[mask] = v[mask] + noise[mask]
        v = np.clip(v, 0.0, 1.0, out=v)
        mutated.append(v.astype(np.float32))
    return mutated


# === Graph sanitation (deterministic + metrics) ===================================

_FACE_PRIORITY = ["FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"]
_FACE_RANK = {f: i for i, f in enumerate(_FACE_PRIORITY)}

def _edge_sort_key(u: int, v: int, data: dict) -> Tuple[int, int]:
    face = data.get("face")
    rank = _FACE_RANK.get(face, 999)
    return (rank, v)

def sanitize_graph_for_mujoco(G_in: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[str, int]]:
    """
    Repair the decoded graph for the MuJoCo constructor:
      - remove nodes with module_type == 'NONE'
      - keep only the weakly-connected component that contains the core
      - BFS arborescence rooted at the core
      - enforce at most one child per (from_node, face)
    Returns repaired graph + metrics dict.
    """
    metrics = {
        "nodes_in": 0, "edges_in": 0,
        "none_nodes_removed": 0,
        "components_removed_nodes": 0,
        "face_collisions_skipped": 0,
        "multi_parent_skipped": 0,
        "unreached_nodes_removed": 0,
        "nodes_out": 0, "edges_out": 0,
    }

    G = G_in.copy()
    metrics["nodes_in"] = G.number_of_nodes()
    metrics["edges_in"] = G.number_of_edges()

    # 1) Drop non-physical modules
    none_nodes = [n for n, d in G.nodes(data=True) if d.get("module_type") == "NONE"]
    metrics["none_nodes_removed"] = len(none_nodes)
    if none_nodes:
        G.remove_nodes_from(none_nodes)

    # If core vanished, return empty graph
    if CORE_NODE not in G:
        H = nx.DiGraph()
        metrics["nodes_out"] = 0
        metrics["edges_out"] = 0
        return H, metrics

    # 2) Keep only the weakly connected component containing the core
    comps = list(nx.weakly_connected_components(G))
    keep = next((c for c in comps if CORE_NODE in c), set())
    drop = set(G.nodes) - set(keep)
    metrics["components_removed_nodes"] = len(drop)
    if drop:
        G.remove_nodes_from(drop)

    if G.number_of_nodes() == 0:
        H = nx.DiGraph()
        return H, metrics

    # 3 & 4) BFS arborescence with one parent per node and one child per face
    H = nx.DiGraph()
    for n, attrs in G.nodes(data=True):
        H.add_node(n, **attrs)

    used_face: Dict[int, Set[str]] = {}
    has_parent: Set[int] = set()
    visited: Set[int] = set([CORE_NODE])
    queue = [CORE_NODE]

    while queue:
        u = queue.pop(0)
        if u not in used_face:
            used_face[u] = set()
        out_edges = list(G.out_edges(u, data=True))
        out_edges.sort(key=lambda e: _edge_sort_key(e[0], e[1], e[2]))

        for _, v, data in out_edges:
            if v not in G:
                continue
            if v in has_parent:
                metrics["multi_parent_skipped"] += 1
                continue
            face = data.get("face")
            if face in used_face[u]:
                metrics["face_collisions_skipped"] += 1
                continue

            H.add_edge(u, v, **data)
            used_face[u].add(face)
            has_parent.add(v)
            if v not in visited:
                visited.add(v)
                queue.append(v)

    keep_nodes = set(visited)
    drop_nodes = set(H.nodes) - keep_nodes
    metrics["unreached_nodes_removed"] = len(drop_nodes)
    if drop_nodes:
        H.remove_nodes_from(drop_nodes)

    metrics["nodes_out"] = H.number_of_nodes()
    metrics["edges_out"] = H.number_of_edges()
    return H, metrics


# === Decode & simulate ============================================================

def decode_graphs(genotype: List[np.ndarray], name: str) -> Tuple[nx.DiGraph, nx.DiGraph, Dict[str, int]]:
    """Decode genotype -> raw HPD graph, sanitize -> return both + metrics, also save to disk."""
    p_mats = GLOBAL_NDE.forward(genotype)
    raw: DiGraph[Any] = GLOBAL_HPD.probability_matrices_to_graph(
        p_mats[0], p_mats[1], p_mats[2]
    )
    san, metrics = sanitize_graph_for_mujoco(raw)
    save_graph_as_json(raw, DATA / "graphs_raw" / f"{name}.json")
    save_graph_as_json(san, DATA / "graphs_sanitized" / f"{name}.json")
    return raw, san, metrics


def construct_core_from_graph(graph: nx.DiGraph) -> Any:
    """Build MuJoCo spec core module from sanitized graph."""
    return construct_mjspec_from_graph(graph)


def simulate_robot_named(core: Any, label: str, duration: int = 6, record_video: bool = True) -> Tuple[float, list]:
    """Simulate a robot, record optional video, return (fitness, history)."""
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)

    # Freeze controller RNG for comparable runs
    rng_state = copy.deepcopy(RNG.bit_generator.state)
    try:
        mode = "video" if record_video else "simple"
        experiment(robot=core, controller=ctrl, mode=mode, duration=duration)
    finally:
        RNG.bit_generator.state = rng_state

    hist_list = tracker.history.get("xpos", [])
    history = hist_list[0] if (hist_list and len(hist_list[0]) > 0) else []
    fit = fitness_function(history) if history else float("-inf")
    console.log(f"[{label}] fitness: {fit:.4f}")
    return fit, history


# === EA structures ================================================================

class Individual:
    def __init__(self, gid: int, geno: List[np.ndarray], origin: str, parents: Tuple[int,int]|None):
        self.id = gid
        self.geno = geno
        self.origin = origin
        self.parents = parents
        self.fitness = float("-inf")
        self.repair: Dict[str,int] = {}
        self.graph_name_base = f"gen{0}_ind{gid}"  # will be updated per-gen

def random_population(n: int, geno_len: int = 64) -> List[List[np.ndarray]]:
    return [random_genotype(geno_len) for _ in range(n)]


# === Main =========================================================================

def main() -> None:
    console.rule("[bold green]A3 EA: NDE-only bodies + random controller")
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    # CSV log
    log_path = DATA / f"run_log_{run_tag}.csv"
    with log_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "run_tag","generation","ind_id","origin","parent_a","parent_b",
            "fitness",
            "nodes_in","edges_in","none_removed","comp_removed_nodes",
            "face_collisions_skipped","multi_parent_skipped","unreached_removed",
            "nodes_out","edges_out",
            "raw_graph_path","san_graph_path","video_recorded"
        ])

        # === Gen 0: random population
        population: List[Individual] = []
        for i, geno in enumerate(random_population(POP_SIZE, 64)):
            population.append(Individual(i, geno, origin="init", parents=None))

        elite: Individual | None = None

        for gen in range(GENERATIONS):
            console.rule(f"[bold magenta]Generation {gen}")

            # Evaluate all (construct graphs, repair, simulate)
            for ind in population:
                ind.graph_name_base = f"gen{gen}_ind{ind.id}"
                raw, san, metrics = decode_graphs(ind.geno, ind.graph_name_base)

                # If sanitized graph is empty, skip sim + give -inf fitness
                video_ok = (not VIDEO_ELITE_ONLY)  # non-elites default no video
                if san.number_of_nodes() == 0 or san.number_of_edges() == 0:
                    ind.fitness = float("-inf")
                    ind.repair = metrics
                    writer.writerow([
                        run_tag, gen, ind.id, ind.origin,
                        "" if not ind.parents else ind.parents[0],
                        "" if not ind.parents else ind.parents[1],
                        ind.fitness,
                        metrics["nodes_in"], metrics["edges_in"],
                        metrics["none_nodes_removed"], metrics["components_removed_nodes"],
                        metrics["face_collisions_skipped"], metrics["multi_parent_skipped"],
                        metrics["unreached_nodes_removed"],
                        metrics["nodes_out"], metrics["edges_out"],
                        str(DATA / "graphs_raw" / f"{ind.graph_name_base}.json"),
                        str(DATA / "graphs_sanitized" / f"{ind.graph_name_base}.json"),
                        False
                    ])
                    continue

                core = construct_core_from_graph(san)

                # record video only for elite later; for now, simple sim
                fit, _ = simulate_robot_named(core, ind.graph_name_base, duration=SIM_DURATION_S, record_video=False)
                ind.fitness = fit
                ind.repair = metrics

                writer.writerow([
                    run_tag, gen, ind.id, ind.origin,
                    "" if not ind.parents else ind.parents[0],
                    "" if not ind.parents else ind.parents[1],
                    ind.fitness,
                    metrics["nodes_in"], metrics["edges_in"],
                    metrics["none_nodes_removed"], metrics["components_removed_nodes"],
                    metrics["face_collisions_skipped"], metrics["multi_parent_skipped"],
                    metrics["unreached_nodes_removed"],
                    metrics["nodes_out"], metrics["edges_out"],
                    str(DATA / "graphs_raw" / f"{ind.graph_name_base}.json"),
                    str(DATA / "graphs_sanitized" / f"{ind.graph_name_base}.json"),
                    False
                ])

            # Pick elite (ties by lowest id)
            elite = max(population, key=lambda ind: (ind.fitness, -ind.id))
            console.log(f"Elite g{gen}: ind {elite.id} | fitness={elite.fitness:.4f} | origin={elite.origin}")

            # Re-simulate elite with video on (once per generation)
            # Use the already-sanitized graph saved to disk
            _, san_elite, _ = decode_graphs(elite.geno, f"gen{gen}_elite_ind{elite.id}")
            if san_elite.number_of_nodes() > 0 and san_elite.number_of_edges() > 0:
                core_elite = construct_core_from_graph(san_elite)
                fit_vid, _ = simulate_robot_named(core_elite, f"gen{gen}_ELITE", duration=SIM_DURATION_S, record_video=True)
                # Log the elite video row as an extra entry for traceability
                writer.writerow([
                    run_tag, gen, elite.id, "elite_video", "", "",
                    fit_vid,
                    "", "", "", "", "", "", "", "", "",
                    str(DATA / "graphs_raw" / f"gen{gen}_elite_ind{elite.id}.json"),
                    str(DATA / "graphs_sanitized" / f"gen{gen}_elite_ind{elite.id}.json"),
                    True
                ])

            # === Next generation (1+19): keep elite, spawn children
            if gen < GENERATIONS - 1:
                next_pop: List[Individual] = []
                # Keep elite as id 0 of next gen (new id namespace per gen)
                next_pop.append(Individual(0, [v.copy() for v in elite.geno], origin="elite", parents=(elite.id, elite.id)))

                # 10 mutations of elite
                for j in range(10):
                    child_g = gaussian_mutation(elite.geno, RNG, sigma=0.05, p=0.10)
                    next_pop.append(Individual(len(next_pop), child_g, origin="mut(elite)", parents=(elite.id, elite.id)))

                # 9 crossovers: elite × random member of current population
                for j in range(9):
                    mate = RNG.integers(0, len(population))
                    while population[mate].id == elite.id and len(population) > 1:
                        mate = RNG.integers(0, len(population))
                    child_g = one_point_crossover(elite.geno, population[mate].geno, RNG)
                    next_pop.append(Individual(len(next_pop), child_g, origin=f"cx(elite,{population[mate].id})", parents=(elite.id, population[mate].id)))

                population = next_pop

    console.rule("[bold cyan]Done")
    console.log("CSV log:", log_path)
    console.log("Raw graphs:", DATA / "graphs_raw")
    console.log("Sanitized graphs:", DATA / "graphs_sanitized")
    console.log("Videos:", VID_DIR)


if __name__ == "__main__":
    main()
