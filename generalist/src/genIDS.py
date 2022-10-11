# %%
import pandas as pd
import numpy as np
from deap import creator,base
import pickle as pkl
from tqdm import tqdm

# %%
import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str

# %%
creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

# %%
df = pd.read_csv("./data/CSV/trainingSpecialists.csv").sort_values(["EA","Enemy","Round","Generation","Individual"])
ids = [0]*df.shape[0]
weights = [0]*df.shape[0]
passedWeights = [0]*df.shape[0]
passedIds = [0]*df.shape[0]

# %%
index = 0
with tqdm(total=df.shape[0]) as iterator:
    for ea in range(1,3):
        for enemy in range(1,9):
            for rnd in range(10):
                for generation in range(21):
                    row = df.loc[(df["EA"] == ea)&(df["Enemy"] == enemy)&(df["Round"] == rnd)&(df["Generation"] == generation) &(df["Individual"] == 0)]
                    with open("./data/checkpoints/S"+ str(int(row["EA"])) + "_" + str(int(row["Enemy"])) + "_" + str(int(row["Round"])) + "/" + str(int(row["Generation"])) + ".pkl","rb") as file_:
                        pop = pkl.load(file_)[0]
                    for individual in range(100):
                        weight = pop[individual]
                        if weight not in passedWeights:
                            weights[index] = weight
                            passedWeights[index] = weight
                            while True:
                                id = get_random_string(16)
                                if id not in passedIds:
                                    ids[index] = id
                                    passedIds[index] = id
                                    break
                        else:
                            ids[index] = ids[weights.index(weight)]
                            weights[index] = weight
                        iterator.update(1)
                        index+=1

# %%
with open("./ids.pkl", "wb") as file_:
    pkl.dump(ids,file_)


