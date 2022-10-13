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
        if "S" in settings["name"].split("_")[0]:
            mode = "no"
        else:
            mode = "yes"
        self.env = Environment(
            multiplemode=mode,
            enemies=self.settings["enemies"],
            level=2,
            speed="fastest",
            sound="off",
            logs="off",
            savelogs="no",
            timeexpire=3000,
            clockprec="low",
            randomini="yes",
            player_controller=player_controller(NEURONS)
        )

        self.toolbox = base.Toolbox()
        creator.create("Fitness", base.Fitness, weights=(1.0,))      
        creator.create("IndividualContainer", list, fitness=creator.Fitness)
        self.toolbox.register("initialSensor", random.uniform, -100, 100)
        self.toolbox.register("individual", tools.initRepeat, creator.IndividualContainer, self.toolbox.initialSensor, n=((self.env.get_num_sensors()+1) * NEURONS + (NEURONS + 1) * 5))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=4, indpb=0.7)
        if self.settings["settings"]["selType"] == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=self.settings["settings"]["tournamentSize"]) 
        elif self.settings["settings"]["selType"] == "roulette":
            self.toolbox.register("select", tools.selRoulette) 

        self.cxpbStep = (self.settings["settings"]["cxpb"]-0.2)/self.settings["settings"]["ngens"]
        self.mutpbStep = (self.settings["settings"]["mutpb"]-0.2)/self.settings["settings"]["ngens"]
        self.cxpb = self.settings["settings"]["cxpb"]
        self.mutpb = 0.2

        def evaluate(individual):
            f,p,e,t = self.env.play(pcont=np.array(individual))
            return f,p,e,t

        self.toolbox.register("evaluate", evaluate)

        self.history = tools.History()
        self.HoF = tools.HallOfFame(5)
        self.dataPerGen = []
        if self.settings["settings"]["popInit"] == "random":
            self.popsize = self.settings["settings"]["popSize"]
            self.population = self.toolbox.population(n=self.popsize)
        elif type(self.settings["settings"]["popInit"]) == type([]):
            self.popNoise = self.settings["settings"]["popSize"] - round((self.settings["settings"]["popSize"]*(1-self.settings["Noise"]))/len(self.settings["enemies"]))*len(self.settings["enemies"])
            self.popsize = self.settings["settings"]["popSize"]
            self.population = []
            for candidate in self.settings["settings"]["popInit"]:
                with open("./data/checkpoints/"+candidate[0], "rb") as cp:
                    checkpoint = pkl.load(cp)
                    self.population.append(checkpoint[0][int(candidate[1])])
            self.population.extend(self.toolbox.population(n=self.popNoise))
        else:
            exit("FATAL ERROR NO POPULATION CREATED")
        self.populations = []

    def generation(self, gen):
        offspring = list(map(self.toolbox.clone, self.toolbox.select(self.population, len(self.population))))
        tempData = [self.dataPerGen[gen][[ind[0] for ind in self.dataPerGen[gen]].index(ind.fitness.values[0])] for ind in offspring]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for index, mutant in enumerate(offspring):
            if random.random() < self.mutpb:
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
        with open("./data/checkpoints/" + self.settings["name"] + "/" + str(gen+1) + ".pkl", "wb") as cp:
            pkl.dump([self.population, self.history, self.HoF, self.dataPerGen], cp)

        print(" Generation Avg:", np.mean([ind.fitness.values[0] for ind in self.population]))

    def train(self):
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
        
        for generation in tqdm(range(self.settings["settings"]["ngens"])):
            self.generation(generation)
            self.cxpb = self.settings["settings"]["cxpb"] - (generation * self.cxpbStep)
            self.mutpb = 0.2 + (generation * self.mutpbStep)

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