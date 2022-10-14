import argparse
parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
parser.add_argument('-EA', '--ea', type=str, required=True, help='Which type to train, either: G or S')
parser.add_argument('-G', '--group', type=str, default="1", help='Which group to train, either: all, 1 or 2 ')
parser.add_argument('-T', '--type', type=str, default="S", help='Which type to train, either: all, S or R')
parser.add_argument('-R', '--rnd', type=int, default=10, help='Number of rounds, must be more than 1')
args = parser.parse_args()
assert args.ea == "G" or args.ea == "S"
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
settings = None
with open("./settings.json", "r") as json_file:
    settings = json.load(json_file)

if "S" in args.ea:
    settings = settings["Specialists1"]
elif "G" in args.ea:
    settings = settings["EA1"]

if args.type == "all":
    types = ["S", "R"]
else:
    types = [args.type]

if args.group == "all":
    groups = ["1", "2"]
else:
    groups = [args.group]

if args.ea == "S":
    for enemy in range(1,9):
        for rnd in range(args.rnd):
            settings["name"] = "S1_" + str(enemy) + "_" + str(rnd)
            settings["enemies"] = [enemy]
            if not os.path.exists("./data/checkpoints/" + settings["name"]):
                os.mkdir("./data/checkpoints/" + settings["name"])
            print("Running:", settings["name"])
            ea = EA(settings)
            ea.train()
            del ea
else:
    for group in groups:
        for type_ in types:
            setting = settings[type_+"_"+group]
            for rnd in range(args.rnd):
                setting["name"] =  "1_" + group + "_" + type_ + "_" + str(rnd)
                if type_ == "S":
                    print("Running on population size:", round(setting["settings"]["popSize"]/len(setting["enemies"]))*len(setting["enemies"]))
                    trainedSpec = pd.read_csv("./data/CSV/trainingSpecialists.csv").sort_values("Fitness", ascending=False).drop_duplicates('ids')
                    population = []
                    for enemy in setting["enemies"]:
                        for rowNR, candidate in trainedSpec.loc[(trainedSpec["EA"] == int(1)) & (trainedSpec["Enemy"] == enemy)].nlargest(int(round(setting["settings"]["popSize"]/len(setting["enemies"]))*len(setting["enemies"])/len(setting["enemies"])),"Fitness").iterrows():
                            population.append(["S" + str(int(candidate["EA"])) + "_" + str(int(candidate["Enemy"])) + "_" + str(int(candidate["Round"])) + "/" + str(int(candidate["Generation"])) + ".pkl", str(int(candidate["Individual"]))])
                    setting["settings"]["popInit"] = population
                else:
                    setting["settings"]["popInit"] = "random"
                if not os.path.exists("./data/checkpoints/" + setting["name"]):
                    os.mkdir("./data/checkpoints/" + setting["name"])
                print("Running:", setting["name"])
                ea = EA(setting)
                ea.train()
                del ea