import sys, os
sys.path.insert(0, 'evoman') 
os.environ["SDL_VIDEODRIVER"] = "dummy"
from environment import Environment
from demo_controller import player_controller, enemy_controller

import random
from deap import tools, base, creator
import numpy as np
from math import ceil
import pandas as pd
from datetime import datetime
import os

##~~Initialization~~##
##Here the enviroment gets setup
n_hidden_neurons = 100
env = Environment(
    # multiplemode="yes",
    # enemies=[1,2,3,4,5,6,7,8],
    multiplemode="no",
    enemies=[6],
    loadplayer="yes",
    loadenemy="yes",
    level=1,
    playermode="ai",
    enemymode="ai",
    speed="fastest"  ,
    inputscoded="no",
    randomini="no",
    sound="off",
    contacthurt="player",
    logs="off",
    savelogs="no",
    timeexpire=3000,
    overturetime=100,
    clockprec="low",
    player_controller=player_controller(n_hidden_neurons)
)

##Here we set the way of the fitness value, 1 is maximize and -1 is minimize,
##The values to fit are: player health (max), enemy health (min) and time (min)
creator.create("Fitness", base.Fitness, weights=(1.0,))      
##Create the container for the individuals
creator.create("IndividualContainer", list, fitness=creator.Fitness)     

##Set some factors like the number of genes
nGenes = (env.get_num_sensors()+1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
geneMin = -1
geneMax = 1
toolbox = base.Toolbox()

##Population##
##Set the values for the population, this is acording to the API of DEAP
toolbox.register("initialSensor", random.uniform, geneMin,geneMax)
toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, toolbox.initialSensor, n=nGenes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

##EA Tools##
##The different functions for the EA
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) 

gamma = 1
alpha = 1
beta = 1.25
celta = 1.004

##This is where the simulation is run, and will return the player health, enemy health and the time (p,e,t) as an array
def evaluate(individual):
    f,p,e,t = env.play(pcont=np.array(individual))
    #f = (gamma*(100-(e**beta))) + (alpha*(100*ceil(p/100))) - (celta**t)
    return f
    
##Set the DEAP function
toolbox.register("evaluate", evaluate)

##Variables##
popSize = 10
population = toolbox.population(n=popSize)
CXPB, MUTPB, NGEN = 0.5, 0.2, 8

##Initial evaluation of the population, this is for running the algo properly
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = (fit,)

##~~The Evolution~~##
fitness_gen = []
for generation in range(NGEN):
    print("Generation: ",generation)
    ##Get some offspring with the tournement selection
    offspring = list(
        map(
            toolbox.clone,
            toolbox.select(population, len(population))
        )
    )
    
    ##Do the crossover, CXPB is the probability of crossingover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
    #Mutate the offspring, the MUTPB is a random chance of mulation
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness, so reevaluate
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit,)

    # The population is entirely replaced by the offspring
    population[:] = offspring
    if generation == 0:
        fitness_gen = [[individual.fitness.values[0] for individual in population]]
    else:
        fitness_gen = fitness_gen + [[individual.fitness.values[0] for individual in population]]
    print('Mean fitness in generation:',generation,'is',np.mean([individual.fitness.values[0] for individual in population]))


##Evaluate the last one with your own eyes :)
env.speed = "normal"
env.logs = "on"
print("NOWGETTING THE BEST ONE!!!! N0.: ",np.argmax([individual.fitness.values[0] for individual in population]))
print("Settings: ", population[np.argmax([individual.fitness.values[0] for individual in population])])
env.play(pcont=np.array(population[np.argmax([individual.fitness.values[0] for individual in population])]))

# Formatting dataframe and Saving results
datim = datetime.now().strftime("%d-%m-%Y_%H:%M")
filename = '_results_log.xlsx'
folder = 'data/log/'
dirpath = os.path.dirname(__file__)
folderpath = folder + datim + filename
filepath = os.path.join(dirpath, folderpath)

fitness_df = pd.DataFrame(fitness_gen).transpose()
fitness_df['individual'] = fitness_df.index
fitness_df = pd.melt(fitness_df, id_vars='individual', value_vars=range(0,NGEN))
fitness_df = fitness_df.sort_values(['variable','individual'])
fitness_df = fitness_df.reset_index(drop=True)
fitness_df = fitness_df.rename(columns={'variable':'generation'})

fitness_df.to_excel(filepath)
