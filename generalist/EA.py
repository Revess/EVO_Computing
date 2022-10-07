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
        self.env = Environment(
            multiplemode="yes",
            enemies=self.settings["enemies"],
            level=2,
            speed="fastest",
            sound="off",
            logs="off",
            savelogs="no",
            timeexpire=3000,
            clockprec="low",
            player_controller=player_controller(NEURONS)
        )

        self.toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights=(1.0,))      
        creator.create("IndividualContainer", list, fitness=creator.Fitness)
        self.toolbox.register("initialSensor", random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, self.toolbox.initialSensor, n=(self.env.get_num_sensors()+1) * NEURONS + (NEURONS + 1) * 5)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        if self.settings["settings"]["cxType"] == "xx":
            self.toolbox.register("mate", tools.cxTwoPoint)
        elif True:
            pass

        if self.settings["settings"]["mutType"] == "xx":
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        elif True:
            pass

        if self.settings["settings"]["selType"] == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=self.settings["settings"]["tournamentSize"]) 
        elif True:
            pass

        def evaluate(individual):
            f,p,e,t = self.env.play(pcont=np.array(individual))
            return f,p,e,t

        self.toolbox.register("evaluate", evaluate)

        self.popsize = self.settings["settings"]["popSize"]
        if self.settings["settings"]["popInit"] == "random":
            self.population = self.toolbox.population(n=self.popsize)
            self.history = tools.History()
            self.HoF = tools.HallOfFame(5)
            self.dataPerGen = []
        elif os.path.exists(self.settings["settings"]["popInit"]):
            with open(self.settings["settings"]["popInit"], "rb") as checkpoint:
                checkpoint = pkl.load(checkpoint)
                #Do something clever here to select the top K or whatever
                self.population = checkpoint[0]
                self.history = checkpoint[1]
                self.HoF = checkpoint[2]
                self.dataPerGen = checkpoint[3]
        self.populations = []

    def generation(self, gen):
        offspring = list(map(self.toolbox.clone, self.toolbox.select(self.population, len(self.population))))
        tempData = [self.dataPerGen[gen][[ind[0] for ind in self.dataPerGen[gen]].index(ind.fitness.values[0])] for ind in offspring]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.settings["settings"]["cxpb"]:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for index, mutant in enumerate(offspring):
            if random.random() < self.settings["settings"]["mutpb"]:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        tempData = [tempData[index] for index, ind in enumerate(offspring) if ind.fitness.valid]
        fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
        self.dataPerGen.append(tempData + fitnesses.copy())
        for index, fitness in enumerate(fitnesses):
            fitnesses[index] = fitness[0]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)

        self.population[:] = offspring
        self.history.update(self.population)
        self.HoF.update(self.population)
        print("Writing Checkpoint")
        with open("./data/checkpoints/" + self.settings["name"] + "/" + str(gen) + ".pkl", "wb") as cp:
            pkl.dump([self.population, self.history, self.HoF, self.dataPerGen], cp)

        print(" Generation Avg:", np.mean([ind.fitness.values[0] for ind in self.population]))

    def train(self):
        if self.settings["settings"]["popInit"] == "random":
            print("Initial Evaluation for random")
            fitnesses = list(tqdm(map(self.toolbox.evaluate, self.population),total=self.settings["settings"]["popSize"]))
            self.dataPerGen.append(fitnesses.copy())
            for index, fitness in enumerate(fitnesses):
                fitnesses[index] = fitness[0]
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = (fit,)
            self.history.update(self.population)
            self.HoF.update(self.population)
        elif os.path.exists(self.settings["settings"]["popInit"]):
            print("Initial Evaluation for fineTuning")
            invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
            tempData = [tempData[index] for index, ind in enumerate(self.population) if ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            self.dataPerGen.append(tempData + fitnesses.copy())
            for index, fitness in enumerate(fitnesses):
                fitnesses[index] = fitness[0]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
            #TODO: maybe here there will be a double generation in the history, maybe set it one back? No idea
            self.history.update(self.population)
            self.HoF.update(self.population)

        for generation in tqdm(range(self.settings["settings"]["ngens"])):
            self.generation(generation)

        self.terminate()

    def terminate(self):
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