##Script meant to run the two EA for 3 different Enemies 10x for 20 generations.
import os
import numpy as np
import pandas as pd
import pickle as pkl

#List for saving the train data of each round
listOfLists = {"evo":[],"enemy":[],"run":[],"generation":[], "individual": [],"fitness":[],"playerHealth":[],"enemyHealth":[],"time":[]}

if not os.path.exists("./dataBackup"):
    os.mkdir("./dataBackup")

for evo in range(2):
    with open("./settings.txt", "r") as file_:
        settings = file_.readlines()
    ##After loading the settings from settings.txt insert the right version, either tournament or roulette
    if evo == 1:
        settings[-2] = "tournament: False\n"
    else:
        settings[-2] = "tournament: True\n"
    with open("./settings.txt", "w") as file_:
        file_.writelines(settings)
    ##Write away the settings
    for enemy in [2,5,8]: #Run each enemy
        for run in range(10): #Run the enemeny 10 times

            #The coming lines are for setting the settings automatically like, name and enemy, the others stay untouched
            name = str(evo) + "_" + str(enemy) + "_" + str(run)
            print("Evo:", evo, "Enemy:", enemy, "Run:", run)
            with open("./settings.txt", "r") as file_:
                settings = file_.readlines()
            settings[-1] = "enemy: " + str(enemy) + "\n"
            settings[-4] = "name: " + name + "\n"
            with open("./settings.txt", "w") as file_:
                file_.writelines(settings)
            
            ##RUN THE EA with the given settings
            os.system("python singlerun.py")

            ##Load the generated data that the EA just trained on
            with open("./data/finetuning/"+name+".pkl","rb") as file_:
                f,p,e,t = pkl.load(file_)

            ##Write the data to the list that is going to be converted to a DF
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

##Close the data by giving it to the dataframe
df = pd.DataFrame.from_dict(listOfLists)
df.to_csv("./trainingResults.csv",index=False)

os.rmdir("./dataBackup")