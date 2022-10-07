import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from src.testSettings import testSettings

from src.demo_controller import player_controller, enemy_controller
import random
from deap import tools, base, creator
import numpy as np
from tqdm import tqdm
import pickle as pkl

class EA():
    def __init__(self):
        testSettings(settings)
        self.settings = settings
        NEURONS = 10
        self.env = Environment(
            multiplemode="yes",
            enemies=self.settings["enemies"],
            level=2,
            speed="normal",
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

        popsize = self.settings["settings"]["popSize"]
        cxpb = self.settings["settings"]["cxpb"]
        mutpb = self.settings["settings"]["mutpb"]
        ngens = self.settings["settings"]["ngens"]
        if self.settings["settings"]["popInit"] == "random":
            population = self.toolbox.population(n=popsize)
            history = tools.History()
            history.update(population)
        elif os.path.exists(self.settings["settings"]["popInit"]):
            with open(self.settings["settings"]["popInit"], "rb") as pop:
                population = pkl.load(pop)

        dataPerGen = []
        populations = []
        HoF = tools.HallOfFame(5)


    def generation(self):
        pass

    def train(self):
        pass