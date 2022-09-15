import random
from deap import tools, base, creator
import numpy as np

##Init values##
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))        #Create the Fitness function.
                #   ^             ^              ^
                # "name",     fitnessfunc,  argument for the fitness func weight -1 is minimization and 1 is maximization
creator.create("Individual", list, fitness=creator.FitnessMin)      #Create an individual person.
                #   ^           ^               ^
                # "name",    storage,   the link to the fitness


IND_SIZE = 10
toolbox = base.Toolbox() #Create a toolbox class, this we will use as our functions for the training.
toolbox.register("attribute", random.random) #Create an attribute that will be registrerd later to the individual
                #     ^             ^
                #   "name",  value of the class
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE) #This individual will have 10 genes (IND_SIZE)
                #     ^             ^                   ^                   ^                  ^
                #   "name",  Call the func n times, Container (class), the function,       n times
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                #     ^               ^            ^            ^
                #   "name",  Call func n times, container (list), func, the arg that is given to the pop will be the populationsize


toolbox.register("mate", tools.cxTwoPoint)
                #   ^           ^
                # "name", give it a 2-point crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) #set the mutation function
toolbox.register("select", tools.selTournament, tournsize=3) #set the selection function

def evaluate(individual):
    return sum(individual)
toolbox.register("evaluate", evaluate) #Set the evation function as a custom function

##The loop doesn't completely work yet##
def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        # print(ind,(fit,),ind.fitness.values)
        ind.fitness.values = (fit,)

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

print(np.min(np.ndarray.flatten(np.array(main()))))