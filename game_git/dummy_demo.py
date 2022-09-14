import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

import random
from deap import tools, base, creator
import numpy as np

n_hidden_neurons = 10

env = Environment(
    multiplemode="no",
    enemies=[8],
    loadplayer="yes",
    loadenemy="yes",
    level=1,
    playermode="ai",
    enemymode="ai",
    speed="normal"  ,
    inputscoded="no",
    randomini="no",
    sound="off",
    contacthurt="player",
    logs="on",
    savelogs="no",
    timeexpire=3000,
    overturetime=100,
    clockprec="low",
    # player_controller=player_controller(n_hidden_neurons),
)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))      
creator.create("IndividualContainer", list, fitness=creator.FitnessMin)     

# nGenes = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
# print(nGenes,env.get_num_sensors())
nGenes = env.get_num_sensors()+1
geneMin = -1
geneMax = 1
toolbox = base.Toolbox()

##Population##
toolbox.register("initialSensor", random.uniform, geneMin,geneMax)
toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, toolbox.initialSensor, n=nGenes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

##EA Tools##
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) 

def evaluate(individual):
    print(individual)
    return sum(individual)
toolbox.register("evaluate", evaluate)

for()
print(env.play())