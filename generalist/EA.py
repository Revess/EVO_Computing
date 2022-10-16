'''
Class for running a single instance of evoman
You can initialize the evoman with editing the settings.json
This can be found in the root together with this file
'''
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
os.environ["SDL_VIDEODRIVER"] = "dummy" 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from src.testSettings import testSettings
from src.demo_controller import player_controller, enemy_controller
import random
from deap import tools, base, creator
import numpy as np
from tqdm import tqdm
import pickle as pkl

class EA():
    def __init__(self, settings):
        testSettings(settings)
        self.settings = settings
        NEURONS = 10
        self.toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights=(1.0,))      
        creator.create("IndividualContainer", list, fitness=creator.Fitness)
        self.toolbox.register("initialSensor", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, self.toolbox.initialSensor, n=((20+1) * NEURONS + (NEURONS + 1) * 5))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=self.settings["settings"]["tournamentSize"]) 

        self.cxpb = self.settings["settings"]["cxpb"]
        self.mutpb = 0.1

        def evaluate(individual):
            totalF = []
            totalP = []
            totalE = []
            totalT = []
            for enemy in self.settings["enemies"]:
                self.env = Environment(
                    multiplemode="no",
                    enemies=[enemy],
                    level=2,
                    speed="fastest",
                    sound="off",
                    logs="off",
                    savelogs="no",
                    timeexpire=3000,
                    clockprec="low",
                    randomini="no",
                    player_controller=player_controller(NEURONS)
                )
                f,p,e,t = self.env.play(pcont=np.array(individual))
                totalF.append(f)
                totalP.append(p)
                totalE.append(e)
                totalT.append(t)

            return self.env.cons_multi(np.array(totalF)),self.env.cons_multi(np.array(totalP)),self.env.cons_multi(np.array(totalE)),self.env.cons_multi(np.array(totalT)), totalF, totalP, totalE, totalT

        self.toolbox.register("evaluate", evaluate)

        ##Setup the population
        self.history = tools.History()
        self.HoF = tools.HallOfFame(5)
        self.dataPerGen = []
        ##Random INI
        if self.settings["settings"]["popInit"] == "random":
            self.popsize = self.settings["settings"]["popSize"]
            self.population = self.toolbox.population(n=self.popsize)
        ##Pretrained model
        elif type(self.settings["settings"]["popInit"]) == type([]):
            self.popsize = self.settings["settings"]["popSize"]
            self.population = []
            for candidate in self.settings["settings"]["popInit"]:
                with open("./data/checkpoints/"+candidate[0], "rb") as cp:
                    checkpoint = pkl.load(cp)
                    self.population.append(checkpoint[0][int(candidate[1])])
        else:
            exit("FATAL ERROR NO POPULATION CREATED")
        self.populations = []

    def generation(self, gen):
        ##Parent selection
        offspring = list(map(self.toolbox.clone, self.toolbox.select(self.population, len(self.population))))
        tempData = [self.dataPerGen[gen][[ind[0] for ind in self.dataPerGen[gen]].index(ind.fitness.values[0])] for ind in offspring]
        ##Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        ##Mutation
        for index, mutant in enumerate(offspring):
            if random.random() < self.mutpb:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        ##Evaluate the survivors
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        tempData = [tempData[index] for index, ind in enumerate(offspring) if ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        self.dataPerGen.append(tempData + fitnesses.copy())
        for index, fitness in enumerate(fitnesses):
            fitnesses[index] = fitness[0]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
        ##Survivor selection
        self.population[:] = offspring
        self.history.update(self.population)
        self.HoF.update(self.population)
        print("Writing Checkpoint")
        with open("./data/checkpoints/" + self.settings["name"] + "/" + str(gen+1) + ".pkl", "wb") as cp:
            pkl.dump([self.population, self.history, self.HoF, self.dataPerGen], cp)

        print("Generation Avg:", np.mean([ind.fitness.values[0] for ind in self.population]))
        print("Generation Max:", np.max([ind.fitness.values[0] for ind in self.population]))

    def train(self):
        ##Initial evaluation of the population
        fitnesses = list(tqdm(map(self.toolbox.evaluate, self.population),total=self.settings["settings"]["popSize"]))
        self.dataPerGen.append(fitnesses.copy())
        for index, fitness in enumerate(fitnesses):
            fitnesses[index] = fitness[0]
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = (fit,)
        self.history.update(self.population)
        self.HoF.update(self.population)

        print("Writing Checkpoint")
        with open("./data/checkpoints/" + self.settings["name"] + "/" + str(0) + ".pkl", "wb") as cp:
            pkl.dump([self.population, self.history, self.HoF, self.dataPerGen], cp)
        
        print("Generation Avg:", np.mean([ind.fitness.values[0] for ind in self.population]))
        print("Generation Max:", np.max([ind.fitness.values[0] for ind in self.population]))
        
        ##Start of the training
        for generation in tqdm(range(self.settings["settings"]["ngens"])):
            self.generation(generation)
            #This takes care of the cross-fade
            self.cxpb = self.settings["settings"]["cxpb"] - (generation * ((self.settings["settings"]["cxpb"]-0.1)/self.settings["settings"]["ngens"]))
            self.mutpb = 0.1 + (generation * ((self.settings["settings"]["mutpb"]-0.1)/self.settings["settings"]["ngens"]))
            self.toolbox.register("select", tools.selTournament, tournsize=round(2+(generation*((self.settings["settings"]["tournamentSize"]-2)/self.settings["settings"]["ngens"])))) 

        self.terminate()

    def terminate(self):
        ##Here the final data will be written
        fitnessPerGen = []
        playerPerGen = []
        enemyPerGen = []
        timePerGen = []
        for generation in self.dataPerGen:
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
        
        print("Writing final population")
        with open("./data/trainedPopulations/" + self.settings["name"] + ".pkl", "wb") as pop:
            pkl.dump(self.population, pop)

        print("Writing train results")
        with open("./data/trainingresults/" + self.settings["name"] + ".pkl", "wb") as data:
            pkl.dump([
                    [[individual[0] for individual in generation] for generation in self.dataPerGen],
                    [[individual[1] for individual in generation] for generation in self.dataPerGen],
                    [[individual[2] for individual in generation] for generation in self.dataPerGen],
                    [[individual[3] for individual in generation] for generation in self.dataPerGen],
                    self.history,
                    self.HoF
                ], data)