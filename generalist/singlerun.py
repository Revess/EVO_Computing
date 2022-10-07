import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from src.testSettings import testSettings

import json
settings = None
with open("./settings.json", "r") as json_file:
    settings = json.load(json_file)
testSettings(settings)

if not os.path.exists("./data"):
    os.mkdir("./data")

if not os.path.exists("./data/trainedPopulations/"):
    os.mkdir("./data/trainedPopulations/")

if not os.path.exists("./data/trainedPopulations/specialists"):
    os.mkdir("./data/trainedPopulations/specialists")

if not os.path.exists("./data/checkpoints"):
    os.mkdir("./data/checkpoints")
if not os.path.exists("./data/checkpoints/" + settings["name"]):
    os.mkdir("./data/checkpoints/" + settings["name"])

os.environ["SDL_VIDEODRIVER"] = "dummy" 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from src.demo_controller import player_controller, enemy_controller
import random
from deap import tools, base, creator
import numpy as np
from tqdm import tqdm
import pickle as pkl
from math import tanh,isnan
from multiprocessing import Pool

NEURONS = 10
env = Environment(
    multiplemode="yes",
    enemies=settings["enemies"],
    level=2,
    speed="normal",
    sound="off",
    logs="off",
    savelogs="no",
    timeexpire=3000,
    clockprec="low",
    player_controller=player_controller(NEURONS)
)
ngenes = (env.get_num_sensors()+1) * NEURONS + (NEURONS + 1) * 5

toolbox = base.Toolbox()
creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)
toolbox.register("initialSensor", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, toolbox.initialSensor, n=ngenes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

if settings["settings"]["cxType"] == "xx":
    toolbox.register("mate", tools.cxTwoPoint)
elif True:
    pass

if settings["settings"]["mutType"] == "xx":
    toolbox.register("mutate", tools.mutGaussian, mu=float(settings[8].split(":")[1].split()[0]), sigma=float(settings[9].split(":")[1].split()[0]), indpb=float(settings[10].split(":")[1].split()[0]))
elif True:
    pass

if settings["settings"]["selType"] == "tournament":
    toolbox.register("select", tools.selTournament, tournsize=settings["settings"]["tournamentSize"]) 
elif True:
    pass

def evaluate(individual):
    f,p,e,t = env.play(pcont=np.array(individual))
    return f,p,e,t

toolbox.register("evaluate", evaluate)

popsize = settings["settings"]["popSize"]
cxpb = settings["settings"]["cxpb"]
mutpb = settings["settings"]["mutpb"]
ngens = settings["settings"]["ngens"]
if settings["settings"]["popInit"] == "random":
    population = toolbox.population(n=popsize)
    history = tools.History()
    history.update(population)
elif os.path.exists(settings["settings"]["popInit"]):
    with open(settings["settings"]["popInit"], "rb") as pop:
        population = pkl.load(pop)

dataPerGen = []
populations = []
HoF = tools.HallOfFame(5)
if __name__ == "__main__":
    print("Initial Evaluation")
    fitnesses = list(map(toolbox.evaluate, population))
    dataPerGen.append(fitnesses.copy())
    for index, fitness in enumerate(fitnesses):
        fitnesses[index] = fitness[0]
    print("Applying the Fitness to the population")
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    for generation in tqdm(range(ngens)):
        offspring = list(map(toolbox.clone,toolbox.select(population, len(population))))
        tempData = [dataPerGen[generation][[ind[0] for ind in dataPerGen[generation]].index(ind.fitness.values[0])] for ind in offspring]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            index+=2

        for index, mutant in enumerate(offspring):
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        tempData = [tempData[index] for index, ind in enumerate(offspring) if ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        dataPerGen.append(tempData + fitnesses.copy())
        for index, fitness in enumerate(fitnesses):
            fitnesses[index] = fitness[0]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        population[:] = offspring
        history.update(population)
        HoF.update(population)
        with open("./data/checkpoints/" + settings["name"] + "/" + generation + ".pkl", "rb") as cp:
            pkl.dump([population, history, HoF, dataPerGen], cp)

        print(" Generation Avg:", np.mean([ind.fitness.values[0] for ind in population]))

    fitnessPerGen = []
    playerPerGen = []
    enemyPerGen = []
    timePerGen = []
    for generation in dataPerGen:
        fitnessGen = []
        playerGen = []
        enemyGen = []
        timeGen = []
        for individual in generation:
            fitnessGen.append(individual[0])
            playerGen.append(individual[1])
            enemyGen.append(individual[2])
            timeGen.append(individual[3])
        fitnessPerGen.append(fitnessGen)
        playerPerGen.append(playerGen)
        enemyPerGen.append(enemyGen)
        timePerGen.append(timeGen)
    
    with open("./data/trainedPopulations/" + settings["name"] + ".pkl", "rb") as pop:
        pkl.dump(population, pop)

    with open("./data/trainingresults/" + settings["name"] + ".pkl", "rb") as data:
        pkl.dump([
                [[individual[0] for individual in generation] for generation in dataPerGen],
                [[individual[1] for individual in generation] for generation in dataPerGen],
                [[individual[2] for individual in generation] for generation in dataPerGen],
                [[individual[3] for individual in generation] for generation in dataPerGen],
                history,
                HoF
            ], data)