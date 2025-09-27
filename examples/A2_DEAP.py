import random
import numpy as np
from deap import base, creator, tools, algorithms

#import simulation and fitness functions
from A2_simulate_and_fitness import evaluate_fitness, save_video

DURATION = 5000 #simulation timesteps, 500 equals 1 sec
NUM_JOINTS = 8 #number of gecko hinges
TARGET = [0.0, -15.0] #target the gecko has to walk towards

#fitness: we want to minimize distance, so weight = -1
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

#generate individual movement seq with dimensions DURATION x NUM_JOINTS
def init_individual():
    return [[random.uniform(-np.pi/2, np.pi/2) for _ in range(NUM_JOINTS)]
            for _ in range(DURATION)]

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#fitness function
def fitness_function(ind):
    movements = np.array(ind)
    return (evaluate_fitness(movements, TARGET),)

toolbox.register("evaluate", fitness_function)

#genetic operators
def mate(ind1, ind2):
    #two-point crossover at timestep level
    cxpoint1 = random.randint(1, DURATION-2)
    cxpoint2 = random.randint(cxpoint1, DURATION-1)
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = \
        ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def mutate(ind, indpb=0.05, sigma=0.1):
    #apply random gaussian noise to joint values
    for t in range(DURATION):
        for j in range(NUM_JOINTS):
            if random.random() < indpb:
                ind[t][j] += random.gauss(0, sigma)
                #clip values to hinge range
                ind[t][j] = max(-np.pi/2, min(np.pi/2, ind[t][j]))
    return ind,

toolbox.register("mate", mate)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

#run the EA
if __name__ == "__main__":
    random.seed(30)
    pop = toolbox.population(n=100) #WE CAN CHANGE POP SIZE

    hof = tools.HallOfFame(1)  #save best individual of each generation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5, mutpb=0.3,
                                   ngen=5, #WE CAN CHANGE NUMBER OF GENERATIONS
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    print("Best fitness found (distance to target):", hof[0].fitness.values[0])
    save_video(np.asarray(hof[0]))
    print("Video of best solution saved in /__videos__")

