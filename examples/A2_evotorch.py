# gecko_evotorch_mlp.py
# EvoTorch + MLP neuroevolution for ARIEL Gecko (MuJoCo)
# - Inputs: [qpos || qvel || sinφ || cosφ]  (qpos/qvel + tiny phase to avoid fixed-point)
# - Outputs: absolute actuator targets (tanh -> ctrl range)
# - Fitness: imported ARIEL tasks (choose at top)
# - Control timing: integer step counter (no float-mod drift)
# - Video: optional; uses project video_renderer

import os, math, random, time
import numpy as np
import torch

import mujoco
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger

# ==== ARIEL tasks (imported, no fallbacks) ====
from ariel.simulation.tasks.gate_learning import xy_displacement, x_speed, y_speed
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.simulation.tasks.turning_in_place import turning_in_place

# ==== ARIEL env & utils ====
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer

# ---------------- Config ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TASK        = "Targeted Locomotion"   # "Targeted Locomotion" | "Gate Learning" | "Turning In Place"
TARGET_XY   = (0.0, -15.0)            # for Targeted Locomotion
SIM_SECONDS = 10.0
CONTROL_HZ  = 25.0
HIDDEN      = 32
VIDEO_DIR   = "./__videos__"
MAKE_VIDEO  = True                    # set False to skip video at the end

# CMA-ES
POPSIZE     = 48
GENERATIONS = 12
SIGMA_INIT  = 0.25

# ------------- Env cache ---------------
G_MODEL = None
G_NQ = G_NV = G_NU = None
G_STEPS_PER_SEC = None
G_ACT_LOW = G_ACT_HIGH = None
G_CORE_GEOM_ID = None

def _find_core_geom_id(model):
    for gid in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm and "core" in nm.lower():
            return gid
    raise RuntimeError("No geom with 'core' found")

def build_env_once():
    global G_MODEL, G_NQ, G_NV, G_NU, G_STEPS_PER_SEC, G_ACT_LOW, G_ACT_HIGH, G_CORE_GEOM_ID
    if G_MODEL is not None: return
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(gecko().spec, spawn_position=[0.0, 0.0, 0.0])
    model = world.spec.compile()
    _ = mujoco.MjData(model)
    G_MODEL = model
    G_NQ, G_NV, G_NU = int(model.nq), int(model.nv), int(model.nu)
    G_STEPS_PER_SEC  = int(round(1.0 / model.opt.timestep))
    G_ACT_LOW  = model.actuator_ctrlrange[:, 0].copy()
    G_ACT_HIGH = model.actuator_ctrlrange[:, 1].copy()
    G_CORE_GEOM_ID = _find_core_geom_id(model)

# ------------- Tiny MLP (NumPy) -----------------
def mlp_param_count(n_in, n_hidden, n_out):
    return n_hidden*n_in + n_hidden + n_out*n_hidden + n_out

def decode_mlp(theta, n_in, n_hidden, n_out):
    p = 0
    W1 = theta[p:p+n_hidden*n_in].reshape(n_hidden, n_in); p += n_hidden*n_in
    b1 = theta[p:p+n_hidden]; p += n_hidden
    W2 = theta[p:p+n_out*n_hidden].reshape(n_out, n_hidden); p += n_out*n_hidden
    b2 = theta[p:p+n_out]; p += n_out
    return W1, b1, W2, b2

def mlp_forward(x, W1, b1, W2, b2):
    h = np.tanh(W1 @ x + b1)
    return np.tanh(W2 @ h + b2)  # [-1,1]^nu

# -------- Control callback (phase + ABS targets + int counter) --------
def build_control_cb(theta, control_hz=CONTROL_HZ):
    build_env_once()
    n_time = 2                              # sinφ, cosφ
    nin = G_NQ + G_NV + n_time
    assert theta.size == mlp_param_count(nin, HIDDEN, G_NU)
    W1, b1, W2, b2 = decode_mlp(theta, nin, HIDDEN, G_NU)

    stride = max(1, int(round(G_STEPS_PER_SEC / control_hz)))
    step = {"i": 0}

    x         = np.empty(nin, dtype=np.float64)
    center    = 0.5 * (G_ACT_HIGH + G_ACT_LOW)
    half_span = 0.5 * (G_ACT_HIGH - G_ACT_LOW)
    last_ctrl = center.copy()

    freq_hz = 1.0
    dphi = 2.0 * math.pi * freq_hz / G_STEPS_PER_SEC
    phi = 0.0

    # diagnostics
    diag = {"clip_hits": 0, "updates": 0}

    def cb(m, d):
        nonlocal phi
        i = step["i"]
        if i % stride == 0:
            # inputs
            x[:G_NQ]          = d.qpos.ravel() / np.pi
            x[G_NQ:G_NQ+G_NV] = np.tanh(0.2 * d.qvel.ravel())
            x[-2] = math.sin(phi); x[-1] = math.cos(phi)

            y = mlp_forward(x, W1, b1, W2, b2)
            target = center + half_span * y
            before = target.copy()
            np.clip(target, G_ACT_LOW, G_ACT_HIGH, out=target)
            # count clips
            diag["clip_hits"] += int(np.any(before != target))
            diag["updates"] += 1

            d.ctrl[:]    = target
            last_ctrl[:] = target
            phi += dphi
        else:
            d.ctrl[:] = last_ctrl
        step["i"] = i + 1

    # expose diagnostics so rollout can read them
    cb._diag = diag
    return cb

# ------------- Rollout + fitness -------------
def rollout_and_fitness(theta, seconds=SIM_SECONDS, control_hz=CONTROL_HZ):
    build_env_once()
    model = G_MODEL
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if data.ctrl is not None: data.ctrl[:] = 0.0

    cb = build_control_cb(theta, control_hz)
    mujoco.set_mjcb_control(cb)

    xy_history = []
    try:
        horizon = int(round(seconds * G_STEPS_PER_SEC))
        for _ in range(horizon):
            mujoco.mj_step(model, data)
            xy = data.geom_xpos[G_CORE_GEOM_ID, :2].copy()
            xy_history.append((float(xy[0]), float(xy[1])))
    finally:
        mujoco.set_mjcb_control(None)

    # fitness (minimize)
    if TASK == "Targeted Locomotion":
        fit = float(distance_to_target(xy_history[-1], TARGET_XY))
    elif TASK == "Gate Learning":
        disp = float(xy_displacement(xy_history))
        xs   = float(x_speed(xy_history, seconds))
        ys   = float(y_speed(xy_history, seconds))
        fit = -(disp + 0.5*xs + 0.5*ys)
    elif TASK == "Turning In Place":
        fit = -float(turning_in_place(xy_history))
    else:
        raise ValueError(f"Unknown TASK: {TASK}")

    # diagnostics: fraction of updates where clipping occurred
    diag = getattr(cb, "_diag", {"clip_hits": 0, "updates": 1})
    clip_frac = diag["clip_hits"] / max(1, diag["updates"])

    return fit, clip_frac

# ------------- Video -------------
def record_video(theta, outdir=VIDEO_DIR, duration=SIM_SECONDS, control_hz=CONTROL_HZ):
    build_env_once()
    os.makedirs(outdir, exist_ok=True)
    model = G_MODEL
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if data.ctrl is not None: data.ctrl[:] = 0.0
    mujoco.set_mjcb_control(build_control_cb(theta, control_hz))
    rec = VideoRecorder(output_folder=outdir)
    try:
        video_renderer(model, data, duration=duration, video_recorder=rec)
    finally:
        mujoco.set_mjcb_control(None)

# ------------- EvoTorch Problem -------------
class GeckoProblem(Problem):
    def __init__(self):
        build_env_once()
        nin = G_NQ + G_NV + 2  # qpos, qvel, sinφ, cosφ
        L = mlp_param_count(nin, HIDDEN, G_NU)
        super().__init__(
            objective_sense="min",
            solution_length=L,
            dtype=torch.float64,
            device="cpu",
        )

    def evaluate(self, solutions: torch.Tensor) -> torch.Tensor:
        # solutions: (batch, L)
        fits = []
        clip_fracs = []  # optional: log saturation frequency
        for sol in solutions:
            theta = sol.detach().cpu().numpy()
            f, clip_frac = rollout_and_fitness(theta)
            fits.append(f)
            clip_fracs.append(clip_frac)
        # Attach diagnostics so logger can see them (optional)
        self.last_clip_frac = float(np.mean(clip_fracs))
        return torch.tensor(fits, dtype=solutions.dtype, device=solutions.device)

# ------------- Main -------------
def main():
    problem = GeckoProblem()
    nin = G_NQ + G_NV + 2
    L = mlp_param_count(nin, HIDDEN, G_NU)

    mean_init = torch.zeros(L, dtype=torch.float64)
    solver = CMAES(problem, popsize=POPSIZE, center_init=mean_init, stdev_init=SIGMA_INIT)
    logger = StdOutLogger(solver, interval=1)

    best_theta = None
    best_fit = float("inf")

    for gen in range(GENERATIONS):
        solver.step()
        pop = solver.population
        # track best
        fits = pop.get_fitnesses().cpu().numpy()
        xs   = pop.access_values().cpu().numpy()
        argmin = int(np.argmin(fits))
        if fits[argmin] < best_fit:
            best_fit = float(fits[argmin])
            best_theta = xs[argmin].copy()

        clip_frac = getattr(problem, "last_clip_frac", float("nan"))
        print(f"[gen {gen}] best={best_fit:.4f}  avg={np.mean(fits):.4f}  clip_frac~{clip_frac:.2f}")

    print(f"[TASK={TASK}] Best fitness: {best_fit:.4f}")

    if MAKE_VIDEO and best_theta is not None:
        record_video(best_theta, outdir=VIDEO_DIR, duration=SIM_SECONDS, control_hz=CONTROL_HZ)

if __name__ == "__main__":
    main()
