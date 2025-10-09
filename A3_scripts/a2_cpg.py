"""
A2 — Gecko locomotion with NA-CPG: learn {phase, amplitudes, w} via Nevergrad
Fitness: distance_to_target((x, y), TARGET_XY) with TARGET_XY = (0.0, -1.0)

Notes:
- We keep the TA-provided NaCPG implementation and only evolve phase, amplitudes, w.
- ha and b are left fixed (as recommended "probably not to adapt").
- Controller callback returns a NumPy 1D array of length nu (critical for MuJoCo).
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import mujoco
import nevergrad as ng
import torch
import matplotlib.pyplot as plt
from mujoco import viewer

# ARIEL imports (keep these exactly as in the template)
from ariel import console
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)
from ariel.utils.runners import simple_runner
from ariel.simulation.tasks.targeted_locomotion import distance_to_target

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- CONFIG ---
RAND_SEED = 3
SIM_SECONDS = 10.0
TARGET_XY = (0.0, -1.0)

# Nevergrad sometimes changes constructor signatures; this helper smooths it over
def make_ng_optimizer(params, budget: int):
    try:
        return ng.optimizers.DifferentialEvolution(parametrization=params, budget=budget)
    except TypeError:
        pass
    try:
        factory = ng.optimizers.DifferentialEvolution()
        return factory(parametrization=params, budget=budget)
    except TypeError:
        pass
    try:
        factory = ng.optimizers.DifferentialEvolution()
        return factory(instrumentation=params, budget=budget)
    except TypeError:
        pass
    return ng.optimizers.NGOpt(parametrization=params, budget=budget)

def show_xpos_history(history: list[list[float]]) -> None:
    arr = np.array(history)
    plt.figure(figsize=(9, 6))
    plt.plot(arr[:, 0], arr[:, 1], "-", label="Path")
    plt.plot(arr[0, 0], arr[0, 1], "go", label="Start")
    plt.plot(arr[-1, 0], arr[-1, 1], "ro", label="End")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True); plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Robot XY trajectory"); plt.legend(); plt.tight_layout()
    plt.show()

def main() -> None:
    # Seeds
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    # Always clear external callbacks first
    mujoco.set_mjcb_control(None)

    # World & body
    world = SimpleFlatWorld()
    gecko_core = gecko()  # DO NOT CHANGE
    world.spawn(gecko_core.spec, position=[0.0, 0.0, 0.0])

    # MuJoCo model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Tracker binds to the gecko's 'core' geom
    tracker = Tracker(mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    # Build NaCPG sized to the number of actuators and with FC adjacency
    nu = int(model.nu)
    adj_dict = create_fully_connected_adjacency(nu)
    na_cpg = NaCPG(adjacency_dict=adj_dict, angle_tracking=False, dt=float(model.opt.timestep))

    # Controller: CRITICAL — return NumPy 1D array, not torch.Tensor
    ctrl = Controller(
        controller_callback_function=lambda m, d: na_cpg.forward(d.time).detach().cpu().numpy(),
        tracker=tracker,
    )
    mujoco.set_mjcb_control(ctrl.set_control)

    # Reset sim
    mujoco.mj_resetData(model, data)

    # ========= Nevergrad instrumentation (ONLY phase, w, amplitudes) =========
    # phase: per-actuator phases in [-2π, 2π]
    # w:     per-actuator frequency-like param; keep in a modest band [0.5, 3.0]
    # amp:   per-actuator output amplitude in [0.1, 1.0]
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(nu,)).set_bounds(-2.0 * math.pi, 2.0 * math.pi),
        w=ng.p.Array(shape=(nu,)).set_bounds(0.5, 3.0),
        amplitudes=ng.p.Array(shape=(nu,)).set_bounds(0.1, 1.0),
    )
    optimizer = make_ng_optimizer(params, budget=300)

    best_fit = float("inf")
    best_params = None

    # Helper: compute distance-to-target from tracker history
    def evaluate_distance() -> float:
        # tracker.history["xpos"] is a list of tracked bodies; take index 0 (core)
        xy_last = tracker.history["xpos"][0][-1]
        loss = distance_to_target((float(xy_last[0]), float(xy_last[1])), TARGET_XY)

    # =================== Optimization loop ===================
    for it in range(optimizer.budget):
        # Ask a candidate
        candidate = optimizer.ask()
        kw = candidate.kwargs

        # Set the three learnable groups in NaCPG
        # (leaving ha and b untouched)
        na_cpg.set_params_by_group("phase", torch.as_tensor(kw["phase"], dtype=torch.float32))
        na_cpg.set_params_by_group("w", torch.as_tensor(kw["w"], dtype=torch.float32))
        na_cpg.set_params_by_group("amplitudes", torch.as_tensor(kw["amplitudes"], dtype=torch.float32))

        # Reset rollout
        tracker.reset()
        mujoco.mj_resetData(model, data)

        # Simulate
        simple_runner(model, data, duration=SIM_SECONDS)

        # Fitness: distance to (0, -1) — smaller is better
        loss = evaluate_distance()

        # Tell optimizer
        optimizer.tell(candidate, loss)

        # Track best
        if loss < best_fit:
            best_fit = loss
            best_params = {k: np.asarray(v, dtype=np.float32) for k, v in kw.items()}

        if (it + 1) % 10 == 0:
            console.log(f"[iter {it+1:03d}] best={best_fit:.4f} last={loss:.4f}")

    console.log(f"[done] best fitness: {best_fit:.4f}")

    # ============== Rerun best and visualize ==============
    if best_params is not None:
        na_cpg.set_params_by_group("phase", torch.from_numpy(best_params["phase"]))
        na_cpg.set_params_by_group("w", torch.from_numpy(best_params["w"]))
        na_cpg.set_params_by_group("amplitudes", torch.from_numpy(best_params["amplitudes"]))
        tracker.reset()
        mujoco.mj_resetData(model, data)
        try:
            viewer.launch(model=model, data=data)  # close window to continue
        finally:
            pass
        show_xpos_history(tracker.history["xpos"][0])

if __name__ == "__main__":
    main()
