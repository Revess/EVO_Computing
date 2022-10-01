import pandas as pd
import matplotlib.pyplot as plt
from plot_functions import get_train_plot
from plot_functions import get_test_boxplot

# Training results
trainres = pd.read_csv('trainingResults.csv')
fig, ax = plt.subplots(2,3)
fig.suptitle('Fitness values over generations per enemy per EA-method', x =0.5, y = 0.94)
for i in range(2):
    if i == 0:
        ev = 0
    elif i == 1:
        ev = 1
    for j in range(3):
        if j == 0:
            en = 2
        elif j == 1:
            en = 5
        elif j == 2:
            en = 8

        get_train_plot(data_frame=trainres, enemy_number=en,evo_number=ev, ax=ax[i][j])

# Save plot
plt.tight_layout()
fig.legend()
fig.set_size_inches(10,6)
plt.show()
#plt.savefig('Fitness over gen per enemy per EA.jpg')

# Test plot
testres = pd.read_csv('testRestults.csv')
get_test_boxplot(testres)
plt.show()
plt.savefig('Boxplot of gain score per EA method.jpg')
