import pickle as pkl
import pandas as pd
import os

trainingDataGeneralists = { "EA": [], "Group": [], "Type": [], "Round": [], "Generation": [], "Individual": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [] }
trainingSpecialists = { "Enemy": [], "Round": [], "Generation": [], "Individual": [], "Fitness": [], "Player Health": [], "Enemy Health": [], "Time": [] }

for file in os.listdir("./data/trainingresults"):
    parseFile = file.split("_")
    if parseFile[0] == "S":
        with open("./data/trainingresults/" + file, "rb") as file_:
            data = pkl.load(file_)
        for generation in range(len(data[0])):
            for ind in range(len(data[0][generation])):
                trainingSpecialists["Enemy"].append(int(parseFile[1]))
                trainingSpecialists["Round"].append(int(parseFile[2]))
                trainingSpecialists["Generation"].append(generation)
                trainingSpecialists["Individual"].append(ind)
                trainingSpecialists["Fitness"].append(data[0][generation][ind])
                trainingSpecialists["Player Health"].append(data[1][generation][ind])
                trainingSpecialists["Enemy Health"].append(data[2][generation][ind])
                trainingSpecialists["Time"].append(data[3][generation][ind])
    else:
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

pd.DataFrame.from_dict(trainingSpecialists).to_csv("./data/CSV/trainingSpecialists.csv", index=False)
pd.DataFrame.from_dict(trainingDataGeneralists).to_csv("./data/CSV/trainingGeneralists.csv", index=False)