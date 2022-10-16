'''
File meant for testing a trained EA
'''
import argparse
parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
parser.add_argument('-EA', '--ea', type=str, required=True, help='Which type to test, either: both, EA1, EA2, bothSE, Specialists1 or Specialists2')
parser.add_argument('-G', '--group', type=str, default="1", help='Which group to test, either: all, 1 or 2 ')
parser.add_argument('-T', '--type', type=str, default="S", help='Which type to test, either: all, S or R')
parser.add_argument('-R', '--rnd', type=int, default=5, help='Number of rounds, must be more than 1')
parser.add_argument('-S', '--slow', help='Slow output', action='store_true')
parser.add_argument('-D', '--display', help='Display output', action='store_true')
args = parser.parse_args()
assert args.ea == "G" or args.ea == "Specialists1" or args.ea == "Specialists2" or args.ea == "bothSE"
assert args.type == "all" or args.type == "S" or args.type == "R"
assert args.group == "all" or args.group == "1" or args.group == "2"
assert args.rnd > 0

if not args.slow:
    SPEED = "fastest"
else:
    SPEED = "normal"

import sys, os
sys.path.insert(0, './evoman') 
if not args.display:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pickle as pkl
import pandas as pd
import numpy as np
from deap import creator, base
from environment import Environment
from src.demo_controller import player_controller, enemy_controller
from tqdm import tqdm
import json

#Initialize
creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

if "Specialists" in args.ea or args.ea == "bothSE":
    ##TEST THE SPECIALISTS
    if args.ea == "Specialists1":
        sEAs = [1]
    elif args.ea == "Specialists2":
        sEAs = [2]
    elif args.ea == "bothSE":
        sEAs = [1,2]

    trainingData = pd.read_csv("./data/CSV/trainingSpecialists.csv").sort_values("Fitness", ascending=False).drop_duplicates('ids')
    testScores = { "EA": [], "Enemy": [], "Training Round": [], "Generation": [], "Individual": [], "Testing Round": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [], "Weights": [] }
    for Sea in sEAs:
        for enemy in range(1,9):
            for rnd in range(trainingData["Round"].max()+1):
                bestPlayer = trainingData.loc[(trainingData["EA"] == Sea) & (trainingData["Enemy"]==enemy) & (trainingData["Round"]==rnd)].nlargest(1,"Fitness").to_dict()
                if len(bestPlayer["Enemy"]) > 0:
                    with open("./data/checkpoints/"+"S"+str(Sea)+"_" + str(list(bestPlayer["Enemy"].values())[0]) + "_" + str(list(bestPlayer["Round"].values())[0])+"/"+str(list(bestPlayer["Generation"].values())[0])+".pkl", "rb") as cp:
                        player = pkl.load(cp)[0][list(bestPlayer["Individual"].values())[0]]
                    print("Testing: ", "S"+str(Sea)+ "_" + str(list(bestPlayer["Enemy"].values())[0]) + "_" + str(list(bestPlayer["Round"].values())[0])+"/"+str(list(bestPlayer["Generation"].values())[0])+".pkl")
                    for r in tqdm(range(args.rnd)):
                        env = Environment(
                            multiplemode="no",
                            enemies=[enemy],
                            level=2,
                            speed=SPEED,
                            sound="off",
                            logs="off",
                            savelogs="no",
                            timeexpire=3000,
                            clockprec="low",
                            randomini="yes",
                            player_controller=player_controller(10)
                        )
                        f,p,e,t = env.play(pcont=np.array(player))
                        testScores["EA"].append(Sea)
                        testScores["Enemy"].append(str(list(bestPlayer["Enemy"].values())[0]))
                        testScores["Training Round"].append(str(list(bestPlayer["Round"].values())[0]))
                        testScores["Generation"].append(str(list(bestPlayer["Generation"].values())[0]))
                        testScores["Individual"].append(str(list(bestPlayer["Individual"].values())[0]))
                        testScores["Testing Round"].append(r)
                        testScores["Fitness"].append(f)
                        testScores["Player Health"].append(p)
                        testScores["Enemy Health"].append(e)
                        testScores["Time"].append(t)
                        testScores["Weights"].append(player)
    if len(testScores["EA"]) > 0:
        if len(sEAs) > 1:
            pd.DataFrame.from_dict(testScores).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
        else: 
            if os.path.exists("./data/CSV/testScoresSpecialists.csv"):
                df = pd.read_csv("./data/CSV/testScoresSpecialists.csv")
                if (1 in list(df["EA"].unique()) and 2 not in list(df["EA"].unique()) and Sea == 1) or (1 not in list(df["EA"].unique()) and 2 in list(df["EA"].unique()) and Sea == 2):
                    pd.DataFrame.from_dict(testScores).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
                elif 1 not in list(df["EA"].unique()) and 2 in list(df["EA"].unique()) and Sea == 1:
                    pd.concat([pd.DataFrame.from_dict(testScores), df]).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
                elif 1 in list(df["EA"].unique()) and 2 not in list(df["EA"].unique()) and Sea == 2:
                    print(df,pd.DataFrame.from_dict(testScores))
                    pd.concat([df, pd.DataFrame.from_dict(testScores)]).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
                elif Sea == 1:
                    pd.concat([pd.DataFrame.from_dict(testScores), df.loc[(df["EA"] == 2)]]).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
                elif Sea == 2:
                    pd.concat([df[(df["EA"] == 1)], pd.DataFrame.from_dict(testScores)]).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)
            else:
                pd.DataFrame.from_dict(testScores).to_csv("./data/CSV/testScoresSpecialists.csv",index=False)

else:
    ##TEST THE GENERALISTS
    trainingData = pd.read_csv("./data/CSV/trainingGeneralists.csv")
    testScores = { "EA": [], "Group": [], "Type": [], "Training Round": [], "Generation": [], "Individual": [], "Enemy": [], "Testing Round": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [], "Weights": [] }
    if args.type == "all":
        types = ["S", "R"]
    else:
        types = [args.type]
    if args.group == "all":
        groups = [1, 2]
    else:
        groups = [int(args.group)]

    for type_ in types:
        for group in groups:
            df = pd.read_csv("./data/CSV/parsedTrainingDataGroup" + str(group) + ".csv")
            if group == 1:
                enemies = list(range(1,9))
            else:
                enemies = [3,5,7,8]
            for rnd in range(trainingData["Round"].max()+1):
                bestPlayer = df.loc[(df["Type"] == type_) & (df["Round"] == rnd)].sort_values(["Relative Beaten", "Sum Player Health", "Sum Time"], ascending=[False, False, True]).iloc[0]
                with open("./data/checkpoints/"+ "1_" + str(bestPlayer["Group"]) + "_" + bestPlayer["Type"] + "_" + str(bestPlayer["Round"]) +"/"+str(bestPlayer["Generation"])+".pkl", "rb") as cp:
                    player = pkl.load(cp)[0][bestPlayer["Individual"]]
                print("Testing: ", "1_" + str(bestPlayer["Group"]) + "_" + bestPlayer["Type"] + "_" + str(bestPlayer["Round"]) +"/"+str(bestPlayer["Generation"])+".pkl")
                for r in tqdm(range(args.rnd)):
                    ##THEST AGAINST EVERY ENEMY
                    for enemy in enemies:
                        env = Environment(
                            multiplemode="no",
                            enemies=[enemy],
                            level=2,
                            speed=SPEED,
                            sound="off",
                            logs="off",
                            savelogs="no",
                            timeexpire=3000,
                            clockprec="low",
                            randomini="no",
                            player_controller=player_controller(10)
                        )
                        f,p,e,t = env.play(pcont=np.array(player))
                        testScores["EA"].append("1")
                        testScores["Group"].append(str(bestPlayer["Group"]))
                        testScores["Type"].append(str(bestPlayer["Type"]))
                        testScores["Training Round"].append(str(bestPlayer["Round"]))
                        testScores["Generation"].append(str(bestPlayer["Generation"]))
                        testScores["Individual"].append(str(bestPlayer["Individual"]))
                        testScores["Enemy"].append(enemy)
                        testScores["Testing Round"].append(r)
                        testScores["Fitness"].append(f)
                        testScores["Player Health"].append(p)
                        testScores["Enemy Health"].append(e)
                        testScores["Time"].append(t)
                        testScores["Weights"].append(player)
    ##Write the CSV away
    pd.DataFrame.from_dict(testScores).to_csv("./data/CSV/testScoresGeneralists.csv",index=False)