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

class SA():
    SETTINGNAMES = ["neurons: ","geneMin: ","geneMax: ","popsize: ","cxpb: ","mutpb: ","ngens: ","npools: ","mu: ","sigma: ","indpb: ","tournsize: ","name: ","finetuning: "]

    def __init__(self, intial, std=0.1, T0=1 ,C=1):
        self.x_old = intial
        self.name_old = str(round(np.random.random()*10000))
        self.y_old = self.runSim(self.x_old, self.name_old)
        self.T0 = T0
        self.std = std
        self.C = C
        self.t = 0

    def proposal(self, x):
        name = str(round(np.random.random()*10000))
        return np.random.multivariate_normal(
            x,
            self.std
        ), self.runSim(x, name), name
        
    def runSim(self, x, name):
        x.append(name)
        x.append("True")
        x = [self.SETTINGNAMES[index]+str(s)+"\n" for index, s in enumerate(x)]
        with open("settings.txt", "w") as file_:
            file_.writelines(x)
        os.system("python singlerun.py")
        with open("./finetuning/"+name+".pkl", "rb") as file_:
            data = pd.DataFrame(pkl.load(file_)).T
        return data[[0]].mean()
    
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
        print("Settings: ", self.x_old)

initial = [
     100, #"neurons":
     -1, #"geneMin":
     1, #"geneMax":
     5, #"popsize":
     0.5, #"cxpb":
     0.2, #"mutpb":
     5, #"ngens":
     5, #"npools":
     0, #"mu":
     1, #"sigma":
     0.1, #"indpb":
     3, #"tournsize":
]

sa = SA(initial)
sa.run(50)