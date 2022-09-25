import sys, os
sys.path.insert(0, 'evoman') 
os.environ["SDL_VIDEODRIVER"] = "dummy"

settings = None
with open("settings.txt", "r") as file_:
    settings = file_.readlines()
npools = int(settings[7].split(":")[1].split()[0])
if npools > 1:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from environment import Environment
from demo_controller import player_controller, enemy_controller
import random
from deap import tools, base, creator
import numpy as np
from tqdm import tqdm
import pickle as pkl
from math import tanh
import threading as td
from multiprocessing import Pool
from plotting import plot
import pandas as pd
from datetime import datetime
from json import tool

neurons = int(settings[0].split(":")[1].split()[0])
env = Environment(
    multiplemode="no",
    enemies=[3],
    level=1,
    speed="fastest",
    sound="off",
    logs="off",
    savelogs="no",
    timeexpire=3000,
    clockprec="low",
    player_controller=player_controller(neurons)
)

# geneMin = int(settings[1].split(":")[1].split()[0])
# geneMax = int(settings[2].split(":")[1].split()[0])
ngenes = (env.get_num_sensors()+1) * neurons + (neurons + 1) * 5

toolbox = base.Toolbox()
creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)
toolbox.register("initialSensor", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, toolbox.initialSensor, n=ngenes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=float(settings[8].split(":")[1].split()[0]), sigma=float(settings[9].split(":")[1].split()[0]), indpb=float(settings[10].split(":")[1].split()[0]))

if settings[-1].split(":")[1].split()[0] == "True":
    toolbox.register("select", tools.selTournament, tournsize=int(settings[11].split(":")[1].split()[0])) 
else:
    toolbox.register("select", tools.selRoulette)

def evaluate(individual):
    f,p,e,t = env.play(pcont=np.array(individual))
    f = -tanh(365/(10**8) * (p**0.15) * (100 - e) * (t - 1000)) * (66-0.33*e) + 0.3*(p**0.15)*(100-e) 
    return f,p,e,t

toolbox.register("evaluate", evaluate)

popsize = int(settings[3].split(":")[1].split()[0])
cxpb = float(settings[4].split(":")[1].split()[0])
mutpb = float(settings[5].split(":")[1].split()[0])
ngens = int(settings[6].split(":")[1].split()[0])
if npools < 1:
    npools = 1
if npools > popsize:
    npools = popsize
population = toolbox.population(n=popsize)


dataPerGen = []
if __name__ == "__main__":
    with Pool(npools) as pool:
        print("Initial Evaluation")
        fitnesses = pool.map(evaluate, population)
        dataPerGen.append(fitnesses.copy())
        for index, fitness in enumerate(fitnesses):
            fitnesses[index] = fitness[0]
        print("Applying the Fitness to the population")
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)

    for generation in tqdm(range(ngens)):
        with Pool(npools) as pool:
            offspring = list(map(toolbox.clone,toolbox.select(population, len(population))))
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
            tempData = [dataPerGen[generation][index] for index,ind in enumerate(offspring) if ind.fitness.valid]
            fitnesses = pool.map(toolbox.evaluate, invalid_ind)
            dataPerGen.append(tempData + fitnesses.copy())
            for index, fitness in enumerate(fitnesses):
                fitnesses[index] = fitness[0]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)

        population[:] = offspring
        print(" Generation Avg:", np.mean([ind.fitness.values[0] for ind in population]))
        # print("Generation Min:", np.min([ind.fitness.values[0] for ind in population]))
        # print("Generation Max:", np.max([ind.fitness.values[0] for ind in population]))

    fitnessPerGen = []
    playerPerGen = []
    enemyPerGen = []
    timePerGen = []
    for index, generation in enumerate(dataPerGen):
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
    
    if settings[13].split(":")[1].split()[0] == 'True':
        if not os.path.exists("./finetuning"):
            os.mkdir("./finetuning")

    with open("./finetuning/"+str(settings[12].split(":")[1].split()[0])+".pkl","wb") as file_:
        pkl.dump([fitnessPerGen,playerPerGen,enemyPerGen,timePerGen], file_)