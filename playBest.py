import sys, os
sys.path.insert(0, 'evoman') 
import pickle as pkl
import pandas as pd
import numpy as np
from deap import creator, base
from environment import Environment
from demo_controller import player_controller, enemy_controller

creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)
env = Environment(
    multiplemode="yes",
    enemies=[4,6,7],
    level=1,
    speed="normal",
    sound="off",
    logs="off",
    savelogs="no",
    timeexpire=3000,
    clockprec="low",
    player_controller=player_controller(150)
)

name = "Run2.pkl"

with open("./finetuning/"+name, "rb") as file_:
    data = pd.DataFrame(pkl.load(file_)).T

with open("./populations/"+name, "rb") as file_:
    best = pkl.load(file_)[np.argmax(data.iloc[:,0].tolist()[-1])]

env.play(pcont=np.array(best))