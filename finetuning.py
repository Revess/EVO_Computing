# import argparse
# parser = argparse.ArgumentParser(description='Train the EA algorithm for EVOMAN')
# parser.add_argument('-n', '--neurons', metavar='N', type=int, default=100, help='Set the number of neurons')
# parser.add_argument('-r', '--range', metavar='[min,max]', type=int, default=[-1,1], nargs='+', help='Set the range of the individuals -r [MIN] [MAX]')
# parser.add_argument('-p', '--popsize', metavar='N', type=int, default=50, help='Population size')
# parser.add_argument('-c', '--cxpb', metavar='F', type=float, default=0.5, help='Crossover probability')
# parser.add_argument('-m', '--mutpb', metavar='F', type=float, default=0.2, help='Mutation probability')
# parser.add_argument('-g', '--ngens', metavar='N', type=int, default=15, help='Number of generations')
# parser.add_argument('-P', '--npools', metavar='N', type=int, default=50, help='Number of generations')

# args = parser.parse_args()

import sys, os
import numpy as np
import pickle as pkl
import pandas as pd
import datetime

class SA():
    SETTINGNAMES = ["neurons: ","geneMin: ","geneMax: ","popsize: ","cxpb: ","mutpb: ","ngens: ","npools: ","mu: ","sigma: ","indpb: ","tournsize: ","name: ","finetuning: ", "tournament: "]
    USETOURNAMENT = True

    def __init__(self, intial, std=0.1, T0=1 ,C=1,prefix=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        self.namePrefix = prefix
        self.stepcount = 0
        self.name_old = self.namePrefix + "_" + str(self.stepcount)
        self.T0 = T0
        self.std = np.array([
                [33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]])
        self.C = C
        self.t = 0
        self.x_old = intial
        self.y_old = self.runSim(self.x_old.copy(), self.name_old)
        self.history = []

    def proposal(self, x):
        name = self.namePrefix + "_" + str(self.stepcount+1)
        new_x = np.random.multivariate_normal(
            x,
            self.std
        ).tolist()
        x_new = []
        for index, elem in enumerate(new_x):
            if index == 0 or index == 3 or index == 6 or index == 7 or index == 11:
                x_new.append(round(elem))
            elif ((index == 4 or index == 5 or index == 8) and elem > 1) or (elem < 0 and index != 8):
                x_new.append(1)
            elif elem < 0 and (index == 4 or index == 5 or index == 8):
                x_new.append(0.1)
            else:
                x_new.append(elem)
        new_x = x_new.copy()
        return new_x, self.runSim(new_x.copy(), name), name
        
    def runSim(self, x, name):
        x.append(name)
        x.append("True")
        x.append(self.USETOURNAMENT)
        x = [self.SETTINGNAMES[index]+str(s)+"\n" for index, s in enumerate(x)]
        with open("settings.txt", "w") as file_:
            file_.writelines(x)
        os.system("python singlerun.py")
        with open("./finetuning/"+name+".pkl", "rb") as file_:
            data = pd.DataFrame(pkl.load(file_)).T
        print(np.mean(data.iloc[:, 0].tolist()[-1]))
        return np.mean(data.iloc[:, 0].tolist()[-1])
    
    def evaluate(self, y_new, y_old, T):
        A = y_new**(1/T)/y_old**(1/T)
        return np.minimum(1.,A)

    def select(self, x_prop, y_prop, name, A):
        if A > np.random.uniform():
            self.x_old = x_prop
            self.y_old = y_prop
            self.name_old = name

    def step(self):
        T = (self.C*np.log(self.t+self.T0))**-1
        self.t = self.t+1
        x_prop, y_prop, name = self.proposal(self.x_old)
        A = self.evaluate(y_prop, self.y_old, T)
        self.select(x_prop, y_prop, name, A)

    def run(self, epochs):
        for epoch in range(epochs):
            print("Epoch: ",epoch+1, "Fitness: ", self.y_old)
            self.step()
            self.stepcount+=1
            self.history.append(self.x_old)
            with open("./history.pkl", "wb") as file_:
                pkl.dump(self.history, file_)
        print("Settings: ", self.x_old)


initial = [
     100, #"neurons":   std: 33
     -1, #"geneMin":    std: 0.1
     1, #"geneMax":     std: 0.1
     50, #"popsize":    std: 33
     0.5, #"cxpb":      std: 0.01
     0.2, #"mutpb":     std: 0.01
     15, #"ngens":      std: 5
     5, #"npools":      std: 33
     0, #"mu":          std: 3
     1, #"sigma":       std: 3
     0.1, #"indpb":     std: 0.01
     3, #"tournsize":   std: 3
]

sa = SA(initial)
sa.run(50)