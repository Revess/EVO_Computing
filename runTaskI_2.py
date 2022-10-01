import os
import numpy as np
import pandas as pd
import pickle as pkl

listOfLists = {"evo":[],"enemy":[],"run":[],"generation":[], "individual": [],"fitness":[],"playerHealth":[],"enemyHealth":[],"time":[]}

for evo in range(1):
    evo = 1
    with open("./settings.txt", "r") as file_:
        settings = file_.readlines()
    if evo == 1:
        settings[-2] = "tournament: False\n"
    else:
        settings[-2] = "tournament: True\n"
    with open("./settings.txt", "w") as file_:
        file_.writelines(settings)
    for enemy in [2,5,8]:
        for run in range(10):
            name = str(evo) + "_" + str(enemy) + "_" + str(run)
            print("Evo:", evo, "Enemy:", enemy, "Run:", run)
            with open("./settings.txt", "r") as file_:
                settings = file_.readlines()
            settings[-1] = "enemy: " + str(enemy) + "\n"
            settings[-4] = "name: " + name + "\n"
            with open("./settings.txt", "w") as file_:
                file_.writelines(settings)
            os.system("python singlerun.py")
            with open("./finetuning/"+name+".pkl","rb") as file_:
                f,p,e,t = pkl.load(file_)
            for gen in range(len(f)):
                for individual in range(len(f[0])):
                    listOfLists["evo"].append(evo)
                    listOfLists["enemy"].append(enemy)
                    listOfLists["run"].append(run)
                    listOfLists["generation"].append(gen)
                    listOfLists["individual"].append(individual)
                    listOfLists["fitness"].append(f[gen][individual])
                    listOfLists["playerHealth"].append(p[gen][individual])
                    listOfLists["enemyHealth"].append(e[gen][individual])
                    listOfLists["time"].append(t[gen][individual])
            df = pd.DataFrame.from_dict(listOfLists)
            df.to_csv("./dataBackup/"+name+".csv",index=False)

df = pd.DataFrame.from_dict(listOfLists)
df.to_csv("./trainingResults.csv",index=False)