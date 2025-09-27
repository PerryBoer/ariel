# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Keep track of data / history (used only in viewer demos)
HISTORY = []

# ---------------------------------------------
# Constants
# ---------------------------------------------
CTRL_MIN, CTRL_MAX = -np.pi/2, np.pi/2

# ---------------------------------------------
# 0) Original random viewer callback (for demo)
# ---------------------------------------------
def random_move(model, data, to_track) -> None:
    """Generate random movements (viewer demo baseline)."""
    num_joints = model.nu
    rand_moves = np.random.uniform(low=CTRL_MIN, high=CTRL_MAX, size=num_joints)
    delta = 0.05
    data.ctrl += rand_moves * delta
    data.ctrl = np.clip(data.ctrl, CTRL_MIN, CTRL_MAX)
    HISTORY.append(to_track[0].xpos.copy())

# ---------------------------------------------
# 1) Genome → controller callback (time-binned)
# ---------------------------------------------
def make_timebinned_callback(genome_bins: np.ndarray, horizon: int, interp: bool = True):
    """
    genome_bins: (BINS, nu) table with actuator targets in [-π/2, π/2]
    horizon:     total simulation steps
    interp:      linear interpolation between bins for smoothness
    """
    B, nu = genome_bins.shape

    def _cb(model, data, to_track):
        # approximate step index from sim time
        dt = model.opt.timestep if hasattr(model, "opt") else 0.002
        if dt <= 0:
            dt = 0.002
        t_step = int(np.floor(data.time / dt))
        t_step = min(max(t_step, 0), max(horizon - 1, 0))

        # map [0..horizon-1] → [0..B-1] (fractional)
        pos = (t_step / max(horizon - 1, 1)) * (B - 1)
        b0 = int(np.floor(pos))
        b1 = min(b0 + 1, B - 1)
        w = pos - b0

        if interp and b1 > b0:
            cmd = (1.0 - w) * genome_bins[b0] + w * genome_bins[b1]
        else:
            cmd = genome_bins[b0]

        if data.ctrl.shape[0] != nu:
            cmd = cmd[: data.ctrl.shape[0]]
        data.ctrl[:] = np.clip(cmd, CTRL_MIN, CTRL_MAX)

    return _cb

# ---------------------------------------------
# 2) Fitness evaluation (headless Δx)
# ---------------------------------------------
def evaluate_genome(genome_bins: np.ndarray, horizon: int, seed: int = 0) -> float:
    """
    Headless rollout; returns forward displacement Δx for this genome.
    """
    mujoco.set_mjcb_control(None)  # important: clear any previous callback

    # Build fresh world & robot (smooth terrain)
    world = SimpleFlatWorld()
    g = gecko()
    # Spawn slightly above ground to avoid clipping
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])

    # Compile & data
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    # Bind the 'core' geom to track forward progress
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    if not to_track:
        raise RuntimeError("Could not find 'core' geom to track.")

    # Install controller for this genome
    cb = make_timebinned_callback(genome_bins, horizon, interp=True)
    mujoco.set_mjcb_control(lambda m, d: cb(m, d, to_track))

    # Initial x
    x0 = float(to_track[0].xpos[0])

    # Step physics headlessly
    for _ in range(horizon):
        mujoco.mj_step(model, data)

    # Final x
    xT = float(to_track[0].xpos[0])

    mujoco.set_mjcb_control(None)  # clear after eval
    return xT - x0

# ---------------------------------------------
# 3) GA utilities: init, selection, crossover, mutation
# ---------------------------------------------
def init_population(pop_size: int, bins: int, nu: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform initialization in actuator range."""
    return rng.uniform(CTRL_MIN, CTRL_MAX, size=(pop_size, bins, nu)).astype(np.float32)

def tournament_pick(fits: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.choice(len(fits), size=k, replace=False)
    return idx[np.argmax(fits[idx])]

def cx_one_point(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A, B = a.ravel(), b.ravel()
    if A.size < 2:
        return a.copy(), b.copy()
    cut = rng.integers(1, A.size)
    C1 = np.concatenate([A[:cut], B[cut:]]).reshape(a.shape)
    C2 = np.concatenate([B[:cut], A[cut:]]).reshape(a.shape)
    return C1, C2

def cx_uniform(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    A, B = a.ravel(), b.ravel()
    mask = rng.random(A.size) < p
    C1, C2 = A.copy(), B.copy()
    C1[mask], C2[mask] = B[mask], A[mask]
    return C1.reshape(a.shape), C2.reshape(a.shape)

def mut_gaussian_clip(x: np.ndarray, pm: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(size=x.shape) < pm
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape)
    y = x.copy()
    y[mask] += noise[mask]
    return np.clip(y, CTRL_MIN, CTRL_MAX)

# ---------------------------------------------
# 4) GA runner (generational, with elitism)
# ---------------------------------------------
def run_ga(
    variant: str = "one_point",  # "one_point" or "uniform"
    bins: int = 16,
    horizon: int = 1500,
    pop_size: int = 30,
    gens: int = 20,
    tourn_k: int = 3,
    cx_rate: float = 0.7,
    pm: float = 0.10,
    sigma: float = 0.08,
    elite_k: int = 2,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    # Probe nu (actuator count) once
    tmp_world = SimpleFlatWorld()
    tmp_world.spawn(gecko().spec, spawn_position=[0, 0, 0.1])
    tmp_model = tmp_world.spec.compile()
    nu = tmp_model.nu

    # Population init
    pop = init_population(pop_size, bins, nu, rng)

    gen_best, gen_mean = [], []
    best_genome, best_fit = None, -np.inf

    for g in range(gens):
        # Evaluate population
        fits = np.zeros(pop_size, dtype=float)
        for i in range(pop_size):
            fits[i] = evaluate_genome(pop[i], horizon=horizon, seed=seed * 10_000 + g * 1_000 + i)

        # Log
        gen_best.append(fits.max())
        gen_mean.append(fits.mean())

        # Track global best
        if fits.max() > best_fit:
            best_fit = fits.max()
            best_genome = pop[fits.argmax()].copy()

        # Elitism
        elite_idx = np.argsort(fits)[-elite_k:]
        elites = pop[elite_idx].copy()

        # Next generation (generational replacement)
        new_pop = [*elites]
        while len(new_pop) < pop_size:
            p1 = pop[tournament_pick(fits, tourn_k, rng)]
            p2 = pop[tournament_pick(fits, tourn_k, rng)]

            # Crossover according to variant
            if rng.random() < cx_rate:
                if variant == "uniform":
                    c1, c2 = cx_uniform(p1, p2, rng, p=0.5)
                else:
                    c1, c2 = cx_one_point(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = mut_gaussian_clip(c1, pm, sigma, rng)
            if len(new_pop) + 1 < pop_size:
                c2 = mut_gaussian_clip(c2, pm, sigma, rng)
                new_pop.extend([c1, c2])
            else:
                new_pop.append(c1)

        pop = np.array(new_pop, dtype=np.float32)
        print(f"[Gen {g+1:02d}] best={gen_best[-1]:.3f}  mean={gen_mean[-1]:.3f}")

    return {
        "best_genome": best_genome,
        "best_fitness": best_fit,
        "gen_best": np.array(gen_best),
        "gen_mean": np.array(gen_mean),
        "nu": nu,
        "bins": bins,
        "horizon": horizon,
    }

# ---------------------------------------------
# 5) Random baseline curve (same representation)
# ---------------------------------------------
def random_baseline_curve(gens: int, bins: int, nu: int, horizon: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(gens):
        genome = rng.uniform(CTRL_MIN, CTRL_MAX, size=(bins, nu)).astype(np.float32)
        vals.append(evaluate_genome(genome, horizon=horizon, seed=rng.integers(1e9)))
    return np.array(vals)

# ---------------------------------------------
# 6) Plotting helpers
# ---------------------------------------------
def plot_curves(results: list[dict], rand_curve: np.ndarray | None, title: str):
    plt.figure(figsize=(9, 5))
    for res in results:
        gb, gm, label = res["gen_best"], res["gen_mean"], res.get("label", "GA")
        x = np.arange(1, len(gb) + 1)
        plt.plot(x, gm, label=f"{label} mean")
        plt.plot(x, gb, "--", label=f"{label} best")
    if rand_curve is not None:
        xR = np.arange(1, len(rand_curve) + 1)
        plt.plot(xR, rand_curve, ":", label="Random baseline")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Δx)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_qpos_history(history:list):
    pos_data = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position'); plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane'); plt.legend(); plt.grid(True)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)
    plt.xlim(-max_range, max_range); plt.ylim(-max_range, max_range)
    plt.show()

# ---------------------------------------------
# 7) Viewer demo (optional)
# ---------------------------------------------
def viewer_demo_random():
    """Run the original random controller in the viewer (for sanity check)."""
    mujoco.set_mjcb_control(None)
    HISTORY.clear()

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)  # type: ignore

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    mujoco.set_mjcb_control(lambda m, d: random_move(m, d, to_track))
    viewer.launch(model=model, data=data)
    if HISTORY:
        show_qpos_history(HISTORY)

# ---------------------------------------------
# 8) Entrypoint: run one GA variant + random curve
# ---------------------------------------------
def main():
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # --- choose crossover variant here: "one_point" or "uniform"
    variant = "one_point"

    ga_out = run_ga(
        variant=variant,
        bins=24,          # controller "architecture" (keep fixed for fairness)
        horizon=6000,     # steps per evaluation
        pop_size=60,
        gens=40,
        tourn_k=3,
        cx_rate=0.85,      # swap 0.7↔0.3 when you set up explorative vs exploitative if desired
        pm=0.10,
        sigma=0.08,
        elite_k=2,
        seed=0,
    )
    ga_out["label"] = f"GA ({variant})"

    # Random baseline curve (same representation)
    rand_curve = random_baseline_curve(
        gens=len(ga_out["gen_best"]),
        bins=ga_out["bins"],
        nu=ga_out["nu"],
        horizon=ga_out["horizon"],
        seed=123,
    )

    plot_curves([ga_out], rand_curve, title="Gecko Neuroevolution (Time-binned Controller)")
    show_qpos_history(HISTORY)

    # Save best controller
    np.save("__best_timebinned_controller__.npy", ga_out["best_genome"])
    print(f"Saved best genome with fitness {ga_out['best_fitness']:.3f} → __best_timebinned_controller__.npy")

if __name__ == "__main__":
    main()
