import pickle as pkl
import pandas as pd
from deap import tools, base, creator
import os

creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

trainingDataGeneralists = { "EA": [], "Group": [], "Type": [], "Round": [], "Generation": [], "Individual": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [] }
trainingSpecialists = { "EA": [], "Enemy": [], "Round": [], "Generation": [], "Individual": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [] }

for file in os.listdir("./data/trainingresults"):
    parseFile = file.split(".")[0].split("_")
    if "S" in parseFile[0]:
        pass
        # with open("./data/trainingresults/" + file, "rb") as file_:
        #     data = pkl.load(file_)
        # for generation in range(len(data[0])):
        #     for ind in range(len(data[0][generation])):
        #         trainingSpecialists["EA"].append(int(parseFile[0][1]))
        #         trainingSpecialists["Enemy"].append(int(parseFile[1]))
        #         trainingSpecialists["Round"].append(int(parseFile[2]))
        #         trainingSpecialists["Generation"].append(generation)
        #         trainingSpecialists["Individual"].append(ind)
        #         trainingSpecialists["Fitness"].append(data[0][generation][ind])
        #         trainingSpecialists["Player Health"].append(data[1][generation][ind])
        #         trainingSpecialists["Enemy Health"].append(data[2][generation][ind])
        #         trainingSpecialists["Time"].append(data[3][generation][ind])
    elif "TOURN" not in parseFile[0]:
        with open("./data/trainingresults/" + file, "rb") as file_:
            data = pkl.load(file_)
        for generation in range(len(data[0])):
            for ind in range(len(data[0][generation])):
                trainingDataGeneralists["EA"].append(int(parseFile[0]))
                trainingDataGeneralists["Group"].append(int(parseFile[1]))
                trainingDataGeneralists["Type"].append(parseFile[2])
                trainingDataGeneralists["Round"].append(int(parseFile[3]))
                trainingDataGeneralists["Generation"].append(generation)
                trainingDataGeneralists["Individual"].append(ind)
                trainingDataGeneralists["Fitness"].append(data[0][generation][ind])
                trainingDataGeneralists["Player Health"].append(data[1][generation][ind])
                trainingDataGeneralists["Enemy Health"].append(data[2][generation][ind])
                trainingDataGeneralists["Time"].append(data[3][generation][ind])

pd.DataFrame.from_dict(trainingDataGeneralists).to_csv("./data/CSV/trainingGeneralists.csv", index=False)