import sys, os
sys.path.insert(0, './evoman') 
from environment import Environment
from src.demo_controller import player_controller, enemy_controller
import pandas as pd
import numpy as np

test = pd.read_csv("./data/CSV/testScoresGeneralists.csv")

dataTest = {"Group":[], "Type": [], "Round": [], "Enemies Beaten": [], "Total Enemies": [], "Sum Player Health": [], "Sum Time": [], "Weights":[]}
for group in [1,2]:
    for type_ in ["S", "R"]:
        for rnd in range(10):
            ind = test.loc[(test["Group"] == group) & (test["Type"] == type_) & (test["Training Round"] == rnd)]
            dataTest["Group"].append(group)
            dataTest["Type"].append(type_)
            dataTest["Round"].append(rnd)
            dataTest["Enemies Beaten"].append(len([e for e in ind["Enemy Health"].to_list() if e == 0]))
            dataTest["Total Enemies"].append(len([e for e in ind["Enemy Health"].to_list()]))
            dataTest["Sum Player Health"].append(sum(ind["Player Health"].to_list()))
            dataTest["Sum Time"].append(sum(ind["Time"].to_list()))
            dataTest["Weights"].append(ind["Weights"].iloc[0])

testingRest = pd.DataFrame.from_dict(dataTest)
veryBest = testingRest.sort_values(["Enemies Beaten", "Sum Player Health", "Sum Time"], ascending=[False, False, True]).iloc[0]
with open("./group52.txt", "w") as file_:
    file_.write(veryBest["Weights"][1:-1].replace(",","\n").replace(" ",""))

for enemy in [1,2,3,4,5,6,7,8]:
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
    print("Score for Enemy", enemy,":", env.play(pcont=np.array([float(w) for w in veryBest["Weights"][1:-1].split(",")])))