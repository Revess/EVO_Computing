import sys, os
sys.path.insert(0, 'evoman') 
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pickle as pkl
import pandas as pd
import numpy as np
from deap import creator, base
from environment import Environment
from demo_controller import player_controller, enemy_controller
from math import tanh
from tqdm import tqdm

creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

df = pd.read_csv("./trainingResults.csv")

bestEnemy2E1 = [0]*10
bestEnemy5E1 = [0]*10
bestEnemy8E1 = [0]*10

for run in range(10):
    tdf = df.loc[(df["evo"] == 0) & (df["enemy"] == 2) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy2E1[run] = individual

for run in range(10):
    tdf = df.loc[(df["evo"] == 0) & (df["enemy"] == 5) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy5E1[run] = individual

for run in range(10):
    tdf = df.loc[(df["evo"] == 0) & (df["enemy"] == 8) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy8E1[run] = individual

bestEnemy2E2 = [0]*10
bestEnemy5E2 = [0]*10
bestEnemy8E2 = [0]*10

for run in range(10):
    tdf = df.loc[(df["evo"] == 1) & (df["enemy"] == 2) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy2E2[run] = individual

for run in range(10):
    tdf = df.loc[(df["evo"] == 1) & (df["enemy"] == 5) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy5E2[run] = individual

for run in range(10):
    tdf = df.loc[(df["evo"] == 1) & (df["enemy"] == 8) & (df["run"] == run) & (df["generation"] == 20)].nlargest(1,"fitness")
    with open("./populations/" + str(tdf["evo"].iloc[0]) + "_" + str(tdf["enemy"].iloc[0]) + "_" + str(tdf["run"].iloc[0]) + ".pkl", "rb") as file_:
        individual = pkl.load(file_)[tdf["individual"].iloc[0]]
    bestEnemy8E2[run] = individual

allRuns = {
    "0_2":bestEnemy2E1,
    "0_5":bestEnemy5E1,
    "0_8":bestEnemy8E1,
    "1_2":bestEnemy2E2,
    "1_5":bestEnemy5E2,
    "1_8":bestEnemy8E2
}

Outcomes = {
    "evo":[],
    "enemy":[],
    "individual":[],
    "round":[],
    "f":[],
    "p":[],
    "e":[],
    "t":[]
}

for run in tqdm(allRuns.keys()):
    env = Environment(
        multiplemode="no",
        enemies=[int(run.split("_")[-1])],
        level=2,
        speed="normal",
        sound="off",
        logs="off",
        savelogs="no",
        timeexpire=3000,
        clockprec="low",
        player_controller=player_controller(100)
    )
    for index, individual in enumerate(allRuns[run]):
        for round in range(1):
            f,p,e,t = env.play(pcont=np.array(individual))
            f = -tanh(365/(10**8) * (max(p,0)**0.15) * (100 - max(e,0)) * (t - 1000)) * (66-0.33*max(e,0)) + 0.3*(max(p,0)**0.15)*(100-max(e,0))
            Outcomes["evo"].append(run.split("_")[0])
            Outcomes["enemy"].append(run.split("_")[-1])
            Outcomes["individual"].append(index)
            Outcomes["round"].append(round)
            Outcomes["f"].append(f)
            Outcomes["p"].append(p)
            Outcomes["e"].append(e)
            Outcomes["t"].append(t)
    
# df = pd.DataFrame.from_dict(Outcomes)
# df.to_csv("./testRestults.csv",index=False)