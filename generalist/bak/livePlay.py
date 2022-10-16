import sys, os
sys.path.insert(0, './evoman') 
from environment import Environment
from src.demo_controller import player_controller, enemy_controller
import pandas as pd
import numpy as np
import pickle as pkl
from deap import base, creator

creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

df = pd.read_csv("./data/CSV/trainingGeneralists.csv")
data = df.loc[(df["EA"] == 1)&(df["EA"] == 1) & (df["Group"] == 1) & (df["Type"] == "S")].nlargest(1,"Fitness")
print(data)

with open("./data/checkpoints/"+ str(data["EA"].to_list()[0]) + "_" + str(data["Group"].to_list()[0]) + "_" + str(data["Type"].to_list()[0]) + "_" + str(data["Round"].to_list()[0]) + "/" + str(data["Generation"].to_list()[0]) + ".pkl", "rb") as file_:
    best = pkl.load(file_)[0][int(data["Individual"].to_list()[0])]

for enemy in range(1,9):
    env = Environment(
        multiplemode="no",
        enemies=[enemy],
        level=2,
        speed="normal",
        sound="off",
        logs="off",
        savelogs="no",
        timeexpire=3000,
        clockprec="low",
        randomini="no",
        player_controller=player_controller(10)
    )
    print(env.play(pcont=np.array(best)))