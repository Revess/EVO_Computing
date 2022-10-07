import argparse
parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
parser.add_argument('-EA', '--ea', type=str, required=True, help='Which type to train, either: both (Both the EA1 and EA2), EA1, EA2 or Specialists')
parser.add_argument('-G', '--group', type=str, default="1", help='Which group to train, either: all, 1 or 2 ')
parser.add_argument('-T', '--type', type=str, default="S", help='Which type to train, either: all, S or R')
parser.add_argument('-R', '--rnd', type=int, default=10, help='Number of rounds, must be more than 1')
args = parser.parse_args()
assert args.ea == "both" or args.ea == "EA1" or args.ea == "EA2" or args.ea == "Specialists"
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

if args.ea == "Specialists":
    settings = [allSettings["Specialists"]]
elif args.ea == "EA1":
    settings = [allSettings["EA1"]]
elif args.ea == "EA2":
    settings = [allSettings["EA2"]]
elif args.ea == "both":
    settings = [allSettings["EA1"], allSettings["EA2"]]

if args.type == "all":
    types = ["S", "R"]
else:
    types = [args.type]

if args.group == "all":
    groups = ["1", "2"]
else:
    groups = [args.group]

for ea, setting in enumerate(settings):
    if args.ea == "Specialists":
        for enemy in range(8):
            for round in range(10):
                setting["name"] = "S_" + str(enemy+1) + "_" + str(round)
                if not os.path.exists("./data/checkpoints/" + setting["name"]):
                    os.mkdir("./data/checkpoints/" + setting["name"])
                print("Running:", setting["name"])
                ea = EA(setting)
                ea.train()
    else:
        for group in groups:
            for type_ in types:
                settingSub = setting[type_+"_"+group]
                for rnd in range(args.rnd):
                    settingSub["name"] = str(ea+1) + "_" + group + "_" + type_ + "_" + str(rnd)
                    if not os.path.exists("./data/checkpoints/" + settingSub["name"]):
                        os.mkdir("./data/checkpoints/" + settingSub["name"])
                    print("Running:", setting["name"])
                    ea = EA(settingSub)
                    ea.train()