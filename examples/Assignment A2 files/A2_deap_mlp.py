# DEAP EA evolving a tiny MLP policy for ARIEL Gecko
# Prints best fitness and saves ~10s video of the best genome.

import random
import numpy as np
from deap import base, creator, tools, algorithms

from A2_deap_mlp_simulate_fitness import (
    mlp_dimensions,
    evaluate_fitness_mlp,
    save_video_mlp,
)

# -----------------
# Config
# -----------------
SEED = 30
random.seed(SEED)
np.random.seed(SEED)

HIDDEN = 32           # keep tiny; 16 or 32 is fine
TARGET = (0.0, -15.0)

# EA settings (small to sanity-check)
MU = 60               # parents
LAMBDA = 120          # offspring per gen
GENS = 10
CX_PB = 0.7
MUT_PB = 0.3
MUT_SIGMA = 0.3     # Gaussian sd per-gene in weight space
INDPB = 0.05          # per-gene mutation probability

# -----------------
# Sizes
# -----------------
dims = mlp_dimensions(hidden=HIDDEN)
L = dims["length"]

# -----------------
# DEAP wiring
# -----------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def init_ind():
    # small random weights around 0
    return creator.Individual(np.random.normal(loc=0.0, scale=0.1, size=L).astype(np.float64))

toolbox.register("individual", init_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_ind(ind):
    theta = np.asarray(ind, dtype=np.float64)
    dist = evaluate_fitness_mlp(theta, target_xy=TARGET, hidden=HIDDEN, control_hz=25.0, video_seconds=10.0)
    return (dist,)

toolbox.register("evaluate", eval_ind)

def mate_blx(ind1, ind2, alpha=0.2):
    # BLX-alpha on real vectors
    x1 = np.asarray(ind1)
    x2 = np.asarray(ind2)
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    span = hi - lo
    low = lo - alpha * span
    high = hi + alpha * span
    child1 = np.random.uniform(low, high)
    child2 = np.random.uniform(low, high)
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2

def mutate_gaussian(ind, mu=0.0, sigma=MUT_SIGMA, indpb=INDPB):
    arr = np.asarray(ind)
    mask = np.random.rand(arr.size) < indpb
    noise = np.random.normal(mu, sigma, size=arr.size)
    arr[mask] += noise[mask]
    ind[:] = arr
    return (ind,)

toolbox.register("mate", mate_blx)
toolbox.register("mutate", mutate_gaussian)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar=lambda a, b: np.array_equal(np.asarray(a), np.asarray(b)))
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=MU, lambda_=LAMBDA,
        cxpb=CX_PB, mutpb=MUT_PB,
        ngen=GENS,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    best = hof[0]
    print(f"Best fitness (final distance): {best.fitness.values[0]:.4f}")

    # Save a ~10s video of the best
    save_video_mlp(np.asarray(best), hidden=HIDDEN, control_hz=25.0, video_seconds=10.0)

if __name__ == "__main__":
    main()
