# A2_deap_from_slides.py
# Minimal slide-style NN + DEAP neuroevolution for the ARIEL Gecko

import math, numpy as np, mujoco, random
from deap import base, creator, tools, algorithms
from mujoco import viewer

# --- ARIEL tasks (imported, as asked) ---
from ariel.simulation.tasks.gate_learning import xy_displacement, x_speed, y_speed
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.simulation.tasks.turning_in_place import turning_in_place

# --- ARIEL env ---
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# ================= Config =================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

TASK        = "Targeted Locomotion"        # "Targeted Locomotion" | "Gate Learning" | "Turning In Place"
TARGET_XY   = (0.0, -15.0)
SIM_SECONDS = 10.0
CONTROL_HZ  = 25.0
HIDDEN      = 16
AMP         = 0.6          # scale of actuator half-span; <1.0 prevents instant saturation
PHASE_HZ    = 1.0          # tiny oscillator to keep things moving

# DEAP (tiny, fast)
MU, LAMBDA, GENS = 30, 60, 8
CX_PB, MUT_PB = 0.7, 0.3
MUT_SIGMA, INDPB = 0.20, 0.10

# ================ Minimal helpers ================
def build_world():
    """Create world, gecko, model, data, and to_track (as in slides)."""
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    body  = gecko()
    world.spawn(body.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data  = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]
    return world, model, data, to_track

def core_geom_id(model):
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm and "core" in nm.lower():
            return gid
    raise RuntimeError("no 'core' geom found")

# ---- tiny MLP (with biases via explicit vectors; still slide-simple) ----
def mlp_param_count(n_in, n_hidden, n_out):
    return (n_in+1)*n_hidden + (n_hidden+1)*n_hidden + (n_hidden+1)*n_out

def decode(theta, n_in, n_hidden, n_out):
    p = 0
    W1 = theta[p:p+(n_in+1)*n_hidden].reshape(n_in+1, n_hidden); p += (n_in+1)*n_hidden
    W2 = theta[p:p+(n_hidden+1)*n_hidden].reshape(n_hidden+1, n_hidden); p += (n_hidden+1)*n_hidden
    W3 = theta[p:p+(n_hidden+1)*n_out].reshape(n_hidden+1, n_out); p += (n_hidden+1)*n_out
    return W1, W2, W3

def ff(x, W1, W2, W3):
    # x includes inputs only; we append bias=1.0 each layer
    x1 = np.concatenate([x, [1.0]])
    h1 = np.tanh(x1 @ W1)
    h1b = np.concatenate([h1, [1.0]])
    h2 = np.tanh(h1b @ W2)
    h2b = np.concatenate([h2, [1.0]])
    y  = np.tanh(h2b @ W3)           # [-1, 1]
    return y

# ============== Controller factory (per rollout) ==============
def make_controller(theta, model, data, control_hz, phase_hz, amp):
    # inputs: qpos || qvel || sinφ || cosφ
    nq, nv, nu = int(model.nq), int(model.nv), int(model.nu)
    n_in = nq + nv + 2
    W1, W2, W3 = decode(theta, n_in, HIDDEN, nu)

    steps_per_sec = int(round(1.0 / model.opt.timestep))
    stride = max(1, int(round(steps_per_sec / control_hz)))
    step = {"i": 0}
    phi  = 0.0
    dphi = 2.0 * math.pi * phase_hz / steps_per_sec

    x = np.empty(n_in, dtype=np.float64)
    center    = 0.5 * (model.actuator_ctrlrange[:,1] + model.actuator_ctrlrange[:,0])
    half_span = 0.5 * (model.actuator_ctrlrange[:,1] - model.actuator_ctrlrange[:,0])
    last_ctrl = center.copy()

    gid_core = core_geom_id(model)
    xy_hist = []

    def cb(m, d):
        nonlocal phi
        i = step["i"]
        if i % stride == 0:
            x[:nq]        = d.qpos.ravel() / np.pi
            x[nq:nq+nv]   = np.tanh(0.2 * d.qvel.ravel())
            x[-2] = math.sin(phi); x[-1] = math.cos(phi)

            y = ff(x, W1, W2, W3)                    # [-1,1]^nu
            target = center + amp * half_span * y    # absolute targets
            np.clip(target, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1], out=target)
            d.ctrl[:]    = target
            last_ctrl[:] = target
            phi += dphi
        else:
            d.ctrl[:] = last_ctrl

        xy = d.geom_xpos[gid_core, :2]
        xy_hist.append((float(xy[0]), float(xy[1])))
        step["i"] = i + 1

    cb._xy_hist = xy_hist
    return cb

# ============== One rollout & fitness =================
def rollout_fitness(theta):
    _, model, data, to_track = build_world()
    ctrl = make_controller(theta, model, data, CONTROL_HZ, PHASE_HZ, AMP)
    mujoco.set_mjcb_control(ctrl)
    try:
        horizon = int(round(SIM_SECONDS / model.opt.timestep))
        for _ in range(horizon):
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_control(None)

    xy = ctrl._xy_hist
    if TASK == "Targeted Locomotion":
        fit = float(distance_to_target(xy[-1], TARGET_XY))
    elif TASK == "Gate Learning":
        fit = -(float(xy_displacement(xy)) + 0.5*float(x_speed(xy, SIM_SECONDS)) + 0.5*float(y_speed(xy, SIM_SECONDS)))
    elif TASK == "Turning In Place":
        fit = -float(turning_in_place(xy))
    else:
        raise ValueError("Unknown TASK")
    return fit

# ===================== DEAP wiring =====================
def main():
    # make one env to read sizes
    _, model, data, _ = build_world()
    n_in = int(model.nq) + int(model.nv) + 2
    L = mlp_param_count(n_in, HIDDEN, int(model.nu))

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(np.random.normal(0.0, 0.2, size=L).astype(np.float64)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (rollout_fitness(np.asarray(ind, dtype=np.float64)),))
    toolbox.register("mate", tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=MUT_SIGMA, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=3)

    hof = tools.HallOfFame(1, similar=lambda a,b: np.array_equal(np.asarray(a), np.asarray(b)))
    stats = tools.Statistics(key=lambda ind: float(ind.fitness.values[0]))
    stats.register("avg", np.mean); stats.register("min", np.min); stats.register("max", np.max)

    pop = toolbox.population(n=MU)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                         cxpb=CX_PB, mutpb=MUT_PB, ngen=GENS,
                                         stats=stats, halloffame=hof, verbose=True)

    best = np.asarray(hof[0], dtype=np.float64)
    print(f"[{TASK}] best fitness: {hof[0].fitness.values[0]:.4f}")

    # Optional: view the best policy once (press ESC to close)
    show = False
    if show:
        _, model, data, _ = build_world()
        cb = make_controller(best, model, data, CONTROL_HZ, PHASE_HZ, AMP)
        mujoco.set_mjcb_control(cb)
        try:
            viewer.launch(model=model, data=data)
        finally:
            mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    main()
