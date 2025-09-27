#!/usr/bin/env python3
"""
Simple population-based GA with DEAP for ARIEL/MuJoCo Gecko on SimpleFlatWorld.

- Genome: flat vector of length L = BINS * nu  (nu discovered from model.nu)
- Decoding: reshape (BINS, nu) -> interpolate to (H, nu); clip to [-pi/2, pi/2]
- Fitness: maximize (- distance(final_core_xy, target_xy))
- Parallel: multiprocessing.Pool; each worker compiles its own MuJoCo world/model once
- Reproducibility: seeds Python, NumPy, and DEAP RNG
- Artifacts: fitness_history.csv and best_genome.npy

Usage (example):
    python ga_gecko_deap.py train --gens 20 --pop_size 30 --bins 16 --horizon 1500 --workers 4 --seed 42 --outdir ./runs

Requirements:
    - deap
    - mujoco
    - ariel (your ARIEL package with SimpleFlatWorld & gecko available)

Author: you + ChatGPT
"""

import os
import json
import math
import time
import argparse
import random
import pathlib
from typing import Tuple, List

import numpy as np

# --- Third-party EA ---
from deap_ga import base, creator, tools

# --- MuJoCo / ARIEL imports ---
import mujoco
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# -----------------------
# Globals for worker use
# -----------------------
G_MODEL = None
G_DATA = None
G_NU = None
G_CORE_BIND = None  # data-bound core geom (for xpos)
G_TARGET_XY = None
G_BINS = None
G_HORIZON = None

CTRL_LOW = -math.pi / 2.0
CTRL_HIGH = math.pi / 2.0


# -----------------------
# Worker initializer
# -----------------------
def _worker_init(target_xy: Tuple[float, float], bins: int, horizon: int, seed: int):
    """
    Each worker compiles its own world/model once and holds MjData, core binding, and constants.
    """
    global G_MODEL, G_DATA, G_NU, G_CORE_BIND, G_TARGET_XY, G_BINS, G_HORIZON

    # Seed NumPy & Python inside worker (determinism for any random usage in worker)
    random.seed(seed + os.getpid())
    np.random.seed(seed + (os.getpid() % 100000))

    # Build world + gecko and compile model
    mujoco.set_mjcb_control(None)  # ensure no global callbacks
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0.0, 0.0, 0.0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Try to find "core" geom by name and bind to data for world coordinates (xpos)
    core_bind = None
    try:
        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        for geom in geoms:
            name = getattr(geom, "name", "") or ""
            if "core" in name.lower():
                core_bind = data.bind(geom)
                break
    except Exception:
        core_bind = None

    if core_bind is None:
        raise RuntimeError("Could not locate a 'core' geom to track in the Gecko model.")

    G_MODEL = model
    G_DATA = data
    G_NU = model.nu
    G_CORE_BIND = core_bind
    G_TARGET_XY = np.array(target_xy, dtype=np.float64)
    G_BINS = int(bins)
    G_HORIZON = int(horizon)


# -----------------------
# Helper: reset state
# -----------------------
def _reset_state():
    """
    Reset MuJoCo data to a clean state (qpos, qvel, act, ctrl) for a new rollout.
    """
    global G_MODEL, G_DATA
    mujoco.mj_resetData(G_MODEL, G_DATA)
    if G_DATA.ctrl is not None:
        G_DATA.ctrl[:] = 0.0


# -----------------------
# Helper: genome -> (H, nu) controls
# -----------------------
def _decode_genome_to_controls(flat: np.ndarray) -> np.ndarray:
    """
    flat (L,) -> (BINS, nu) -> linearly interpolate to (HORIZON, nu), clamp to [CTRL_LOW, CTRL_HIGH]
    """
    global G_NU, G_BINS, G_HORIZON

    L = flat.shape[0]
    expected = G_BINS * G_NU
    if L != expected:
        raise ValueError(f"Genome length {L} != BINS({G_BINS}) * nu({G_NU}) = {expected}")

    waypoints = flat.reshape(G_BINS, G_NU)

    # Interpolate each actuator over horizon steps
    # Build time indices: waypoints at t=0,...,BINS-1; we need values at t in linspace [0, BINS-1] with H points
    t_src = np.arange(G_BINS, dtype=np.float64)
    t_dst = np.linspace(0.0, G_BINS - 1.0, G_HORIZON, dtype=np.float64)

    controls = np.empty((G_HORIZON, G_NU), dtype=np.float64)
    for j in range(G_NU):
        controls[:, j] = np.interp(t_dst, t_src, waypoints[:, j])

    # Clip to actuator bounds
    np.clip(controls, CTRL_LOW, CTRL_HIGH, out=controls)
    return controls


# -----------------------
# Evaluation (per individual)
# -----------------------
def _evaluate_individual(flat_genome: np.ndarray) -> Tuple[float]:
    """
    Returns a 1-tuple fitness: (-distance_to_target,)
    """
    global G_MODEL, G_DATA, G_CORE_BIND, G_TARGET_XY

    # Decode genome to control sequence
    controls = _decode_genome_to_controls(flat_genome)

    # Reset sim state
    _reset_state()

    # Rollout
    for t in range(G_HORIZON):
        G_DATA.ctrl[:] = controls[t]
        mujoco.mj_step(G_MODEL, G_DATA)

    # Final XY position of core
    final_xy = np.array(G_CORE_BIND.xpos[:2], dtype=np.float64)

    # Distance to target
    dist = float(np.linalg.norm(final_xy - G_TARGET_XY))

    # DEAP maximizes, so we return negative distance
    return (-dist,)


# -----------------------
# Utility: one-shot model build to get nu
# -----------------------
def discover_nu_once() -> int:
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    g = gecko()
    world.spawn(g.spec, spawn_position=[0.0, 0.0, 0.0])
    model = world.spec.compile()
    return int(model.nu)


# -----------------------
# GA Registration
# -----------------------
def build_toolbox(L: int, pm: float, sigma: float, cx_rate: float, tourn_k: int, seed: int):
    """
    Build a minimal DEAP toolbox with:
      - creator.FitnessMax
      - creator.Individual
      - initialization Uniform[-pi/2, pi/2]
      - cxOnePoint
      - mutGaussian + clip
      - selTournament
    """
    # Recreate creators safely (idempotent)
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Attribute generator: uniform in actuator bounds
    def attr_float():
        return rng.uniform(CTRL_LOW, CTRL_HIGH)

    toolbox.register("individual",
                     tools.initIterate,
                     creator.Individual,
                     lambda: np.array([attr_float() for _ in range(L)], dtype=np.float64))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxOnePoint)

    def _mut_gaussian(individual: np.ndarray):
        # Gaussian mutation per gene with prob pm, then clip
        mask = np_rng.random(individual.shape[0]) < pm
        individual[mask] += np_rng.normal(loc=0.0, scale=sigma, size=mask.sum())
        np.clip(individual, CTRL_LOW, CTRL_HIGH, out=individual)
        return (individual,)

    toolbox.register("mutate", _mut_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=tourn_k)

    # Parameters stored for use in algorithm
    toolbox.cx_rate = cx_rate
    toolbox.pm = pm
    toolbox.sigma = sigma
    toolbox.rng = rng

    return toolbox


# -----------------------
# GA loop (simple generational with elitism)
# -----------------------
def run_ga(args):
    # Global seeding
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Discover nu in main process to size genomes
    nu = discover_nu_once()
    L = args.bins * nu

    # Build toolbox
    toolbox = build_toolbox(
        L=L,
        pm=args.pm,
        sigma=args.sigma,
        cx_rate=args.cx_rate,
        tourn_k=args.tourn_k,
        seed=args.seed
    )

    # Output dirs
    outdir = pathlib.Path(args.outdir).absolute()
    run_dir = outdir / f"algo=ga_simple" / f"seed={args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = {
        "seed": args.seed,
        "bins": args.bins,
        "horizon": args.horizon,
        "pop_size": args.pop_size,
        "gens": args.gens,
        "cx_rate": args.cx_rate,
        "pm": args.pm,
        "sigma": args.sigma,
        "elite_k": args.elite_k,
        "tourn_k": args.tourn_k,
        "workers": args.workers,
        "target_xy": [args.target_x, args.target_y],
        "nu": nu,
        "genome_length": L,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Build population
    pop = toolbox.population(n=args.pop_size)

    # Multiprocessing pool with per-process MuJoCo init
    import multiprocessing as mp
    pool = mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=((args.target_x, args.target_y), args.bins, args.horizon, args.seed)
    )
    toolbox.register("map", pool.map)

    # Evaluation wrapper for DEAP (expects 1-tuple)
    def eval_ind(ind):
        return _evaluate_individual(np.asarray(ind, dtype=np.float64))

    # Evaluate initial population
    fitnesses = list(toolbox.map(eval_ind, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Logging
    hist_rows = []
    start = time.time()

    # Evolution loop
    for gen in range(args.gens):
        gen_start = time.time()

        # Elitism: keep best elite_k
        elite = tools.selBest(pop, k=args.elite_k)

        # Select parents
        offspring = toolbox.select(pop, len(pop) - args.elite_k)
        offspring = list(map(np.copy, offspring))  # copy ndarray payloads

        # Variation
        # Mating
        for i in range(1, len(offspring), 2):
            if random.random() < toolbox.cx_rate:
                tools.cxOnePoint(offspring[i - 1], offspring[i])

        # Mutation
        for ind in offspring:
            toolbox.mutate(ind)

        # New population
        pop = elite + offspring

        # Evaluate invalid fitness
        invalid = [ind for ind in pop if not ind.fitness.valid]
        if invalid:
            fits = list(toolbox.map(eval_ind, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

        # Stats
        fits = np.array([ind.fitness.values[0] for ind in pop], dtype=np.float64)
        best = float(np.max(fits))
        mean = float(np.mean(fits))
        std = float(np.std(fits))

        hist_rows.append({
            "gen": gen,
            "best": best,
            "mean": mean,
            "std": std,
            "time_s": float(time.time() - gen_start),
        })

        # Print concise progress
        print(f"[Gen {gen:03d}] best={best:.4f} mean={mean:.4f} std={std:.4f}")

    total_time = time.time() - start
    print(f"Done. Total time: {total_time:.2f}s")

    # Save history
    import csv
    with open(run_dir / "fitness_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "best", "mean", "std", "time_s"])
        w.writeheader()
        w.writerows(hist_rows)

    # Save best genome
    best_ind = tools.selBest(pop, k=1)[0]
    np.save(run_dir / "best_genome.npy", np.asarray(best_ind, dtype=np.float64))

    # Close pool
    pool.close()
    pool.join()


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Simple DEAP GA for ARIEL Gecko towards target.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Run GA training.")
    t.add_argument("--bins", type=int, default=16, help="Number of control waypoints (BINS).")
    t.add_argument("--horizon", type=int, default=1500, help="Rollout steps per evaluation.")
    t.add_argument("--pop_size", type=int, default=30, help="Population size.")
    t.add_argument("--gens", type=int, default=20, help="Generations.")
    t.add_argument("--cx_rate", type=float, default=0.7, help="Crossover probability (per pair).")
    t.add_argument("--pm", type=float, default=0.10, help="Mutation per-gene probability.")
    t.add_argument("--sigma", type=float, default=0.08, help="Gaussian mutation std.")
    t.add_argument("--elite_k", type=int, default=2, help="Elites carried over each gen.")
    t.add_argument("--tourn_k", type=int, default=3, help="Tournament size.")
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="Parallel workers.")
    t.add_argument("--seed", type=int, default=42, help="Random seed.")
    t.add_argument("--target_x", type=float, default=0.0, help="Target X (world).")
    t.add_argument("--target_y", type=float, default=-15.0, help="Target Y (world).")
    t.add_argument("--outdir", type=str, default="./runs", help="Output directory.")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        run_ga(args)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
