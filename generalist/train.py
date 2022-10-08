import argparse
parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
parser.add_argument('-EA', '--ea', type=str, required=True, help='Which type to train, either: bothEA (Both the EA1 and EA2), EA1, EA2, bothSE, Specialists1 or Specialists2')
parser.add_argument('-G', '--group', type=str, default="1", help='Which group to train, either: all, 1 or 2 ')
parser.add_argument('-T', '--type', type=str, default="S", help='Which type to train, either: all, S or R')
parser.add_argument('-R', '--rnd', type=int, default=10, help='Number of rounds, must be more than 1')
args = parser.parse_args()
assert args.ea == "bothEA" or args.ea == "EA1" or args.ea == "EA2" or args.ea == "Specialists1" or args.ea == "Specialists2" or args.ea == "bothSE"
assert args.type == "all" or args.type == "S" or args.type == "R"
assert args.group == "all" or args.group == "1" or args.group == "2"
assert args.rnd > 0

import os
os.environ["SDL_VIDEODRIVER"] = "dummy" 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

if not os.path.exists("./data"):
    os.mkdir("./data")

if not os.path.exists("./data/trainedPopulations/"):
    os.mkdir("./data/trainedPopulations/")

if not os.path.exists("./data/trainingresults"):
    os.mkdir("./data/trainingresults")

if not os.path.exists("./data/checkpoints"):
    os.mkdir("./data/checkpoints")

import numpy as np
import pandas as pd
import pickle as pkl
from EA import EA
import json
allSettings = None
with open("./settings.json", "r") as json_file:
    allSettings = json.load(json_file)
if "Specialists" in args.ea:
    settings = [allSettings[args.ea]]
elif args.ea == "bothSE":
    settings = [allSettings["Specialists1"], allSettings["Specialists2"]]
elif args.ea == "EA1":
    settings = [allSettings["EA1"]]
elif args.ea == "EA2":
    settings = [allSettings["EA2"]]
elif args.ea == "bothEA":
    settings = [allSettings["EA1"], allSettings["EA2"]]

if args.type == "all":
    types = ["S", "R"]
else:
    types = [args.type]

if args.group == "all":
    groups = ["1", "2"]
else:
    groups = [args.group]

for eaRun, setting in enumerate(settings):
    if "Specialists" in args.ea or args.ea == "bothSE":
        for enemy in range(1,9):
            for rnd in range(args.rnd):
                if "1" in args.ea:
                    setting["name"] = "S1_" + str(enemy+1) + "_" + str(rnd)
                elif "2" in args.ea:
                    setting["name"] = "S2_" + str(enemy+1) + "_" + str(rnd)
                else:
                    setting["name"] = "S"+str(eaRun+1)+"_" + str(enemy) + "_" + str(rnd)
                setting["enemies"] = [enemy]
                if not os.path.exists("./data/checkpoints/" + setting["name"]):
                    os.mkdir("./data/checkpoints/" + setting["name"])
                print("Running:", setting["name"])
                ea = EA(setting)
                ea.train()
                del ea
    else:
        for group in groups:
            for type_ in types:
                settingSub = setting[type_+"_"+group]
                for rnd in range(args.rnd):
                    if "1" in args.ea:
                        eaName = "1"
                    elif "2" in args.ea:
                        eaName = "2"
                    else:
                        eaName = str(eaRun+1)
                    settingSub["name"] = eaName + "_" + group + "_" + type_ + "_" + str(rnd)
                    if type_ == "S":
                        print("Running on size:", round(settingSub["settings"]["popSize"]/len(settingSub["enemies"]))*len(settingSub["enemies"]))
                        settingSub["settings"]["popSize"] = round(settingSub["settings"]["popSize"]/len(settingSub["enemies"]))*len(settingSub["enemies"])
                        trainedSpec = pd.read_csv("./data/CSV/trainingSpecialists.csv")
                        population = []
                        for enemy in settingSub["enemies"]:
                            for rowNR, candidate in trainedSpec.loc[(trainedSpec["EA"] == int(eaName)) & (trainedSpec["Enemy"] == enemy)].nlargest(int(settingSub["settings"]["popSize"]/len(settingSub["enemies"])),"Fitness").iterrows():
                                population.append(["S" + str(int(candidate["EA"])) + "_" + str(int(candidate["Enemy"])) + "_" + str(int(candidate["Round"])) + "/" + str(int(candidate["Generation"])) + ".pkl", str(int(candidate["Individual"]))])
                        settingSub["settings"]["popInit"] = population
                    else:
                        settingSub["settings"]["popInit"] = "random"
                    if not os.path.exists("./data/checkpoints/" + settingSub["name"]):
                        os.mkdir("./data/checkpoints/" + settingSub["name"])
                    print("Running:", settingSub["name"])
                    ea = EA(settingSub)
                    ea.train()
                    del ea