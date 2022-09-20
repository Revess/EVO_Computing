import sys, os
sys.path.insert(0, 'evoman') 
# os.environ["SDL_VIDEODRIVER"] = "dummy" #Uncomment this line to run headless
from environment import Environment
from demo_controller import player_controller, enemy_controller
import random
from deap import tools, base, creator
import numpy as np
from math import ceil

import argparse
parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
parser.add_argument('-n', '--neurons', metavar='N', type=int, default=100, help='Set the number of neurons')
parser.add_argument('-r', '--range', metavar='[min,max]', type=int, default=[-1,1], nargs='+', help='Set the range of the individuals -r [MIN] [MAX]')
parser.add_argument('-p', '--popsize', metavar='N', type=int, default=10, help='Population size')
parser.add_argument('-c', '--cxpb', metavar='F', type=float, default=0.5, help='Crossover probability')
parser.add_argument('-m', '--mutpb', metavar='F', type=float, default=0.2, help='Mutation probability')
parser.add_argument('-g', '--ngens', metavar='N', type=int, default=15, help='Number of generations')

args = parser.parse_args()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~VAR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
N_HIDDEN_NEURONS = args.neurons
GENE_VAL_MIN = args.range[0]
GENE_VAL_MAX = args.range[1]
toolbox = base.Toolbox()

#Alternative Fitness Variables
GAMMA = 1
ALPHA = 1
BETA = 1.25
DELTA = 1.0015

#Run functions
POPSIZE = args.popsize
CXPB = args.cxpb
MUTPB = args.mutpb
NGEN = args.ngens
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

##~~~~~~~~~~~~~~~~~~~~~~~~Initialization~~~~~~~~~~~~~~~~~~~~~~~##
##Here the enviroment gets setup
##Here edit the enviroment of the AI, no need to edit this we'll do this next week when training.
env = Environment(
    # multiplemode="yes",
    # enemies=[1,2,3,4,5,6,7,8],
    multiplemode="no",
    enemies=[3],
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
    player_controller=player_controller(N_HIDDEN_NEURONS)
)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~VAR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
NGENES = (env.get_num_sensors()+1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

##Here we set the way of the fitness value, 1 is maximize and -1 is minimize,
##The values to fit are: player health (max), enemy health (min) and time (min)
creator.create("Fitness", base.Fitness, weights=(1.0,))      
##Create the container for the individuals
creator.create("IndividualContainer", list, fitness=creator.Fitness)

##~~~~~~~~~~~~~~~~~~~Population Functions~~~~~~~~~~~~~~~~~~~~~~##
##Set the values for the population, this is acording to the API of DEAP
toolbox.register("initialSensor", random.uniform, GENE_VAL_MIN, GENE_VAL_MAX)
toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, toolbox.initialSensor, n=NGENES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

##~~~~~~~~~~~~~~~~~~~~~~~~~~EA Tools~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##The different functions for the EA
toolbox.register("mate", tools.cxTwoPoint)
##EDIT HERE THE CROSSOVER FUNCTION, edit tools.cxTwoPoint to your function
##WHEN CHANGING THIS FUNCTION CHECK THAT IT STILL WORKS WITH 2 CHILDREN!!!! OTHERWISE EDIT THIS INSIDE THE TRAINING LOOP

toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
##EDIT HERE THE MUTATION FUNCTION, edit tools.mutGaussian to your function

toolbox.register("select", tools.selTournament, tournsize=3) 
##EDIT HERE THE SELECTION FUNCTION, edit tools.selTournament to your function

##This is where the simulation is run, and will return the player health, enemy health and the time (p,e,t) as an array
##Here the evaluation function is run, add a new line below the env.play and set f to be a different fitness function.
def evaluate(individual):
    f,p,e,t = env.play(pcont=np.array(individual))
    # f = (GAMMA*(100-(e**BETA))) + (ALPHA*(100*ceil(p/100))) - (DELTA**t)
    return f

##Set the DEAP function
toolbox.register("evaluate", evaluate)
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

population = toolbox.population(n=POPSIZE)

##Initial evaluation of the population, this is for running the algo properly
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = (fit,)

##~~~~~~~~~~~~~~~~~~~~~~~~The Evolution~~~~~~~~~~~~~~~~~~~~~~~~##
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
    ##In case of a different crossover (for example with 3 offspring you will need to edit this!!!!)
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
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


##Evaluate the last one with your own eyes :)

###HERE THE CODE OF YARI SHOULD INSERT!!!, THUS THE PLOTS ETC
env.speed = "normal"
env.logs = "on"
print("NOWGETTING THE BEST ONE!!!! N0.: ",np.argmax([individual.fitness.values[0] for individual in population]))
print("Settings: ", population[np.argmax([individual.fitness.values[0] for individual in population])])
while True:
    env.play(pcont=np.array(population[np.argmax([individual.fitness.values[0] for individual in population])]))