# A2_super_simple_ea.py
# Smallest-possible neuroevolution-style baseline:
# - Controller: per-joint sine waves (global amplitude, per-joint phase)
# - EA: (1+lambda) hill-climber (no external libs)
# - Fitness: ARIEL import (pick task at top)
# - Exactly the slide pattern: to_track + set_mjcb_control

import math, numpy as np, mujoco, random
from mujoco import viewer

# ---- ARIEL fitness (imported) ----
from ariel.simulation.tasks.gate_learning import xy_displacement, x_speed, y_speed
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.simulation.tasks.turning_in_place import turning_in_place

# ---- ARIEL env ----
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# ------------------- config -------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

TASK        = "Targeted Locomotion"     # "Targeted Locomotion" | "Gate Learning" | "Turning In Place"
TARGET_XY   = (0.0, -15.0)

SIM_SECONDS = 10.0
FREQ_HZ     = 1.0                       # sine frequency
ALPHA       = 0.15                      # smoothing toward target (prevents instant curl-up)
A_MIN, A_MAX = 0.05, 0.80               # allowed amplitude scale

LAMBDA      = 12                        # children per generation
GENS        = 12                        # generations
STEP_SIGMA_PHASE = 0.35                 # mutation scale (radians)
STEP_SIGMA_AMP   = 0.10                 # mutation scale for amplitude

# -------------- slide-style env helpers --------------
def build_world():
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    body  = gecko()
    world.spawn(body.spec, spawn_position=[0, 0, 0])

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(g) for g in geoms if "core" in g.name]

    # find core geom id (for fast XY logging)
    gid_core = None
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm and "core" in nm.lower():
            gid_core = gid; break
    if gid_core is None:
        raise RuntimeError("no 'core' geom found")

    return world, model, data, to_track, gid_core

# -------------- tiniest controller possible --------------
def make_sine_controller(params, model, gid_core):
    """
    params = [phase_0, ..., phase_{nu-1}, amplitude_scale]
    output = absolute setpoints (smoothed) within actuator range
    """
    nu = int(model.nu)
    phases = np.asarray(params[:nu], dtype=np.float64)
    amp    = float(np.clip(params[-1], A_MIN, A_MAX))

    center    = 0.5 * (model.actuator_ctrlrange[:,1] + model.actuator_ctrlrange[:,0])
    half_span = 0.5 * (model.actuator_ctrlrange[:,1] - model.actuator_ctrlrange[:,0])

    diag = {"updates":0, "clip_hits":0}
    xy_hist = []

    def cb(m, d):
        t = d.time
        # per-joint sine with individual phases, shared freq & amplitude
        y = np.sin(2.0 * math.pi * FREQ_HZ * t + phases)      # [-1,1]^nu
        target = center + amp * half_span * y                 # absolute setpoints

        before = target.copy()
        np.clip(target, model.actuator_ctrlrange[:,0], model.actuator_ctrlrange[:,1], out=target)
        diag["clip_hits"] += int(np.any(before != target))
        diag["updates"]   += 1

        # smooth toward the target (prevents immediate saturation)
        d.ctrl[:] = d.ctrl + ALPHA * (target - d.ctrl)

        # log XY of the core
        xy = d.geom_xpos[gid_core, :2]
        xy_hist.append((float(xy[0]), float(xy[1])))

    cb._diag = diag
    cb._xy_hist = xy_hist
    return cb

# -------------- one rollout + fitness --------------
def evaluate_params(params):
    _, model, data, to_track, gid_core = build_world()

    cb = make_sine_controller(params, model, gid_core)
    mujoco.set_mjcb_control(cb)
    try:
        horizon = int(round(SIM_SECONDS / model.opt.timestep))
        for _ in range(horizon):
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_control(None)

    xy = cb._xy_hist
    if TASK == "Targeted Locomotion":
        fit = float(distance_to_target(xy[-1], TARGET_XY))  # minimize
    elif TASK == "Gate Learning":
        fit = -(float(xy_displacement(xy)) + 0.5*float(x_speed(xy, SIM_SECONDS)) + 0.5*float(y_speed(xy, SIM_SECONDS)))
    elif TASK == "Turning In Place":
        fit = -float(turning_in_place(xy))
    else:
        raise ValueError("Unknown TASK")

    # quick debug
    u, c = cb._diag["updates"], cb._diag["clip_hits"]
    print(f"updates={u}, clip_frac≈{c/max(1,u):.2f}, fit={fit:.4f}")
    return fit

# -------------- tiny (1+λ) hill-climber --------------
def evolve():
    # read sizes once to create initial params
    _, model, _, _, _ = build_world()
    nu = int(model.nu)

    # params: per-joint phase ∈ [-π, π], amplitude ∈ [A_MIN, A_MAX]
    best = np.zeros(nu + 1, dtype=np.float64)
    best[:nu] = np.random.uniform(-math.pi, math.pi, size=nu)
    best[-1]  = np.random.uniform(0.4, 0.7)   # decent starting amplitude

    best_fit = evaluate_params(best)

    for g in range(GENS):
        # sample λ children by Gaussian mutation
        kids, fits = [], []
        for _ in range(LAMBDA):
            child = best.copy()
            child[:nu] += np.random.normal(0.0, STEP_SIGMA_PHASE, size=nu)
            child[:nu] = (child[:nu] + math.pi) % (2*math.pi) - math.pi  # wrap phases to [-π, π]
            child[-1]  = np.clip(child[-1] + np.random.normal(0.0, STEP_SIGMA_AMP), A_MIN, A_MAX)
            kids.append(child)
            fits.append(evaluate_params(child))

        # pick best of parent+children (μ=1)
        idx = int(np.argmin([best_fit] + fits))
        if idx == 0:
            print(f"[gen {g}] keep parent  | best={best_fit:.4f}")
        else:
            best = kids[idx-1]; best_fit = fits[idx-1]
            print(f"[gen {g}] new champion | best={best_fit:.4f}")

    return best, best_fit

def main():
    best, best_fit = evolve()
    print(f"[{TASK}] final best fitness: {best_fit:.4f}")

    # Optional: watch best in the viewer
    watch = True
    if watch:
        _, model, data, _, gid_core = build_world()
        cb = make_sine_controller(best, model, gid_core)
        mujoco.set_mjcb_control(cb)
        try:
            viewer.launch(model=model, data=data)
        finally:
            mujoco.set_mjcb_control(None)

if __name__ == "__main__":
    main()
