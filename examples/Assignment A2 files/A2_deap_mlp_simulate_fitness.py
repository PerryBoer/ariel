import os
import time
import math
import numpy as np
import mujoco

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer

# ==============================
# Cached env globals
# ==============================
G_MODEL = None
G_NQ = None
G_NV = None
G_NU = None
G_STEPS_PER_SEC = None
G_ACT_LOW = None
G_ACT_HIGH = None
G_CORE_GEOM_ID = None

def _find_core_geom_id(model):
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and "core" in name.lower():
            return gid
    raise RuntimeError("Could not find a geom with 'core' in its name")

def _build_env_once():
    global G_MODEL, G_NQ, G_NV, G_NU, G_STEPS_PER_SEC, G_ACT_LOW, G_ACT_HIGH, G_CORE_GEOM_ID
    if G_MODEL is not None:
        return
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0.0, 0.0, 0.1])
    model = world.spec.compile()

    G_MODEL = model
    G_NQ, G_NV, G_NU = int(model.nq), int(model.nv), int(model.nu)
    G_STEPS_PER_SEC = int(round(1.0 / model.opt.timestep))
    G_ACT_LOW  = model.actuator_ctrlrange[:, 0].copy()
    G_ACT_HIGH = model.actuator_ctrlrange[:, 1].copy()

    data = mujoco.MjData(model)
    G_CORE_GEOM_ID = _find_core_geom_id(model)

# ==============================
# MLP utilities
# ==============================
def _mlp_param_count(n_in, n_hidden, n_out):
    return n_hidden * n_in + n_hidden + n_out * n_hidden + n_out

def _decode_mlp(theta, n_in, n_hidden, n_out):
    p = 0
    W1 = theta[p : p + n_hidden * n_in].reshape(n_hidden, n_in); p += n_hidden * n_in
    b1 = theta[p : p + n_hidden]; p += n_hidden
    W2 = theta[p : p + n_out * n_hidden].reshape(n_out, n_hidden); p += n_out * n_hidden
    b2 = theta[p : p + n_out]; p += n_out
    return W1, b1, W2, b2

def _mlp_forward(x, W1, b1, W2, b2):
    h = np.tanh(W1 @ x + b1)
    y = np.tanh(W2 @ h + b2)  # [-1, 1]
    return y

def _scale_to_ctrlrange(u_norm):
    return G_ACT_LOW + 0.5 * (u_norm + 1.0) * (G_ACT_HIGH - G_ACT_LOW)

# ==============================
# Public: dimensions for DEAP
# ==============================
def mlp_dimensions(hidden: int = 32):
    _build_env_once()
    # inputs: qpos/π (nq) + tanh(0.2*qvel) (nv) + dxy/15 (2) + sin,cos (2)
    nin = G_NQ + G_NV + 4
    nout = G_NU
    length = _mlp_param_count(nin, hidden, nout)
    return {"nin": nin, "nout": nout, "hidden": hidden, "length": length, "steps_per_sec": G_STEPS_PER_SEC}

# ==============================
# Core rollout with LEAKY-ABSOLUTE control (anti-saturation)
# ==============================
def _rollout_theta(theta: np.ndarray,
                   target_xy=(0.0, -15.0),
                   hidden: int = 32,
                   control_hz: float = 25.0,
                   video_seconds: float = 10.0,
                   record_moves: bool = False):
    """
    Control law:
      u_norm = MLP(state) in [-1,1]
      target = scale_to_ctrlrange(u_norm)
      ctrl  <- ctrl + beta * (target - ctrl)    # leaky absolute control
    This avoids one-shot saturation and yields sustained motion if the target varies.
    """
    _build_env_once()
    model = G_MODEL
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if data.ctrl is not None:
        data.ctrl[:] = 0.0

    nin = G_NQ + G_NV + 4
    W1, b1, W2, b2 = _decode_mlp(theta, nin, hidden, G_NU)

    horizon = int(round(video_seconds * G_STEPS_PER_SEC))
    control_stride = max(1, int(round(G_STEPS_PER_SEC / control_hz)))
    omega = 2.0 * math.pi * 1.0  # 1 Hz clock
    BETA = 0.1                   # approach rate per control tick (0<beta<=1)

    x = np.empty(nin, dtype=np.float64)
    ctrl = np.zeros(G_NU, dtype=np.float64)
    moves = [] if record_moves else None
    tgt = np.asarray(target_xy, dtype=np.float64)

    for t in range(horizon):
        if t % control_stride == 0:
            # --- features ---
            x[:G_NQ] = data.qpos.ravel() / math.pi
            x[G_NQ:G_NQ+G_NV] = np.tanh(0.2 * data.qvel.ravel())
            xy = np.array(data.geom_xpos[G_CORE_GEOM_ID, :2], dtype=np.float64)
            dxy = (tgt - xy) / 15.0
            x[G_NQ+G_NV:G_NQ+G_NV+2] = dxy
            t_sec = (t // control_stride) / control_hz
            ang = omega * t_sec
            x[-2] = math.sin(ang)
            x[-1] = math.cos(ang)

            # policy -> absolute target via leaky integrator
            u_norm = _mlp_forward(x, W1, b1, W2, b2)
            target = _scale_to_ctrlrange(u_norm)
            ctrl += BETA * (target - ctrl)
            # safety clip
            np.clip(ctrl, G_ACT_LOW, G_ACT_HIGH, out=ctrl)

        data.ctrl[:] = ctrl
        if record_moves:
            moves.append(ctrl.copy())
        mujoco.mj_step(model, data)

    final_xy = np.array(data.geom_xpos[G_CORE_GEOM_ID, :2], dtype=np.float64)
    dist = float(np.linalg.norm(final_xy - tgt))
    return dist, moves

# ==============================
# Public: fitness & video
# ==============================
def evaluate_fitness_mlp(theta: np.ndarray,
                         target_xy=(0.0, -15.0),
                         hidden: int = 32,
                         control_hz: float = 25.0,
                         video_seconds: float = 10.0) -> float:
    dist, _ = _rollout_theta(theta, target_xy, hidden, control_hz, video_seconds, record_moves=False)
    return dist

def save_video_mlp(theta: np.ndarray,
                   output_folder: str = "./__videos__",
                   filename_prefix: str = "gecko_mlp_best",
                   target_xy=(0.0, -15.0),
                   hidden: int = 32,
                   control_hz: float = 25.0,
                   video_seconds: float = 10.0):
    _build_env_once()
    dist, moves = _rollout_theta(theta, target_xy, hidden, control_hz, video_seconds, record_moves=True)
    print(f"[video] distance during recording: {dist:.4f}  steps={len(moves)}")

    # Replay via callback (your pattern)
    mujoco.set_mjcb_control(None)
    model = G_MODEL
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if data.ctrl is not None:
        data.ctrl[:] = 0.0

    step = {"i": 0}
    def controller(m, d):
        i = step["i"]
        if i < len(moves):
            d.ctrl[:] = moves[i]
        step["i"] = i + 1

    mujoco.set_mjcb_control(controller)

    os.makedirs(output_folder, exist_ok=True)
    vr = VideoRecorder(output_folder=output_folder)
    try:
        video_renderer(model, data, duration=int(round(video_seconds)), video_recorder=vr)
    finally:
        try:
            vr.close()
        except Exception:
            pass
        mujoco.set_mjcb_control(None)

    print(f"[video] Saved: {os.path.join(output_folder, 'gecko_mlp_best_*.mp4')}")
