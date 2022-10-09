#from tkinter.tix import COLUMN
from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import src.plotFunctions as pf
import numpy as np

folder = "./data/CSV/"
file = "trainingSpecialists.csv"

trainres = pd.read_csv(folder + file)

x_label = "Enemy"
y_label = "Fitness"
enemies = 5
EA = 1
top = 50
trainres = trainres[trainres["EA"] == EA].drop("EA", axis=1)

lowest_fitness = 100
highest_fitness = 0
averages = [0 for a in range(1, 1 + enemies, 1)]
print(averages)
for x in range(1, 1 + enemies, 1):
    temptrainres = trainres[trainres["Enemy"] == x]
    best = temptrainres["Fitness"].drop_duplicates().sort_values(ascending=False)[0:top]
    to_plot = temptrainres.loc[best.index]
    min_value = to_plot.iloc[top - 1]["Fitness"]
    max_value = to_plot.iloc[0]["Fitness"]

    if lowest_fitness > min_value:
        lowest_fitness = min_value
    if highest_fitness < max_value:
        highest_fitness = max_value
    
    mean = to_plot["Fitness"].mean()
    averages[x - 1] = mean
    
    plt.boxplot(to_plot["Fitness"], positions=[x])
    plt.scatter(to_plot[x_label], to_plot[y_label], label = x_label + str(x), alpha=0.3)
    

plt.plot(range(1, 1 + enemies, 1), averages, alpha=0.9, label="Average")
plt.title("Comparison of top {k} specialists by enemy".format(k=top))
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.ylim(lowest_fitness - 0.5, highest_fitness + 0.5)
plt.xlim(1-0.5, enemies + 0.5)
plt.legend()
plt.show()