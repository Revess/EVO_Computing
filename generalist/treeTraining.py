import os
os.environ["SDL_VIDEODRIVER"] = "dummy" 
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import pandas as pd
import pickle as pkl
from EA import EA
import json
from deap import creator, base

settings = {
    "name": "1_1_S_0",
    "settings": {
        "popSize": 96*2,
        "popInit": "random",
        "cxpb": 0.7,
        "mutpb": 0.4,
        "selType": "tournament",
        "tournamentSize": 64,
        "ngens": 20
    },
    "Noise": 0
}
tournament = {
    0: {
        0: [2,8],
        1: [1,7],
        2: [3,6],
        3: [4,5],
    },
    1: {
        0: [2,8,1,7],
        1: [3,6,4,5],
    }, 
    2: {
        0: [2,8,1,7,3,6,4,5]
    }
}

for eaRun in range(1,3):
    for rnd in range(10):
        for tier in list(tournament.keys()):
            for match in tournament[tier].values():
                population = []
                settings["enemies"] = match
                settings["name"] = "TOURN_" + str(eaRun) + "_" + str(rnd) + "_" + str(tier) + "_" + "-".join([str(e) for e in match])
                if tier == 0:
                    trainedSpec = pd.read_csv("./data/CSV/trainingSpecialists.csv").sort_values("Fitness", ascending=False).drop_duplicates('ids')
                    for enemy in match:
                        candidates = trainedSpec.loc[(trainedSpec["EA"] == eaRun) & (trainedSpec["Enemy"] == enemy)].nlargest(int(settings["settings"]["popSize"]/len(match)), "Fitness")
                        for rowNum, candidate in candidates.iterrows():
                            population.append(["S"+str(int(candidate["EA"])) + "_" + str(int(candidate["Enemy"])) + "_" + str(int(candidate["Round"])) + "/" + str(int(candidate["Generation"]))+".pkl",str(int(candidate["Individual"]))])
                elif tier > 0:
                    trainedSpec = pd.read_csv("./data/CSV/tier"+ str(tier-1) +".csv")
                    enemy = "-".join([str(e) for e in match[int(len(match)/2):]])
                    candidates = trainedSpec.loc[(trainedSpec["EA"] == eaRun) & (trainedSpec["Enemy"] == enemy)].nlargest(int(settings["settings"]["popSize"]/len(match)), "Fitness")
                    for rowNum, candidate in candidates.iterrows():
                        population.append(["TOURN_" + str(eaRun) + "_" + str(rnd) + "_" + str(tier-1) + "_" + "-".join([str(e) for e in match[int(len(match)/2):]]) + "/" + str(int(candidate["Generation"])) + ".pkl",str(int(candidate["Individual"]))])
                    
                    enemy = "-".join([str(e) for e in match[:int(len(match)/2)]])
                    candidates = trainedSpec.loc[(trainedSpec["EA"] == eaRun) & (trainedSpec["Enemy"] == enemy)].nlargest(int(settings["settings"]["popSize"]/len(match)), "Fitness")
                    for rowNum, candidate in candidates.iterrows():
                        population.append(["TOURN_" + str(eaRun) + "_" + str(rnd) + "_" + str(tier-1) + "_" + "-".join([str(e) for e in match[:int(len(match)/2)]]) + "/" + str(int(candidate["Generation"])) + ".pkl",str(int(candidate["Individual"]))])
                if tier == 2:
                    settings["enemies"] = [1,2,3,4,5,6,7,8]
                settings["settings"]["popInit"] = population
                if not os.path.exists("./data/checkpoints/" + settings["name"]):
                    os.mkdir("./data/checkpoints/" + settings["name"])
                print("Running:", settings["name"])
                ea = EA(settings)
                ea.train()
                del ea
        
            trainingTier = { "EA": [], "Round": [], "Tier": [], "Enemy": [], "Generation": [], "Individual": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [] }
            for file in os.listdir("./data/trainingresults"):
                parseFile = file.split(".")[0].split("_")
                if parseFile[0] == "TOURN" and parseFile[3] == str(tier):
                    with open("./data/trainingresults/" + file, "rb") as file_:
                        data = pkl.load(file_)
                    for generation in range(len(data[0])):
                        for ind in range(len(data[0][generation])):
                            trainingTier["EA"].append(int(parseFile[1]))
                            trainingTier["Round"].append(int(parseFile[2]))
                            trainingTier["Tier"].append(parseFile[3])
                            trainingTier["Enemy"].append(parseFile[4])
                            trainingTier["Generation"].append(generation)
                            trainingTier["Individual"].append(ind)
                            trainingTier["Fitness"].append(data[0][generation][ind])
                            trainingTier["Player Health"].append(data[1][generation][ind])
                            trainingTier["Enemy Health"].append(data[2][generation][ind])
                            trainingTier["Time"].append(data[3][generation][ind])
            pd.DataFrame.from_dict(trainingTier).to_csv("./data/CSV/tier"+ str(tier) +".csv", index=False)