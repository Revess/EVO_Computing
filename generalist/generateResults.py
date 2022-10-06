import pandas as pd
import matplotlib.pyplot as plt
import src.plotFunctions as pf
import seaborn as sns

# Training results
trainres = pd.read_csv('./data/trainingResultsAltFitness.csv')
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

        pf.get_train_plot(data_frame=trainres, enemy_number=en,evo_number=ev, ax=ax[i][j])

plt.tight_layout()
fig.legend()
fig.set_size_inches(10,6)
plt.show()
#plt.savefig('Fitness over gen per enemy per EA.jpg') #Uncomment and comment plt.show() to save an image of the plot

# Testing plots

# Gain results
sns.set(font_scale = 1.2, style = 'ticks')
testres = pd.read_csv('./data/testRestultsAltFitness.csv')
pf.get_test_boxplot(testres)
plt.show()
#plt.savefig('Boxplot of gain score per EA method.jpg') #Uncomment and comment plt.show() to save an image of the plot

# Baseline comparison
testres_bas = pd.read_csv('./data/testRestultsBaseline.csv')
pf.get_bas_alt_scatter(alt_df=testres, base_df=testres_bas)
plt.show()
#plt.savefig('Scatterplot of fitness functions.jpg') #Uncomment and comment plt.show() to save an image of the plot

pf.get_bas_alt_ph_boxplot(alt_df=testres, base_df=testres_bas)
plt.show()
#plt.savefig('Boxplot of player health fitness functions.jpg') #Uncomment and comment plt.show() to save an image of the plot

pf.get_bas_alt_vt_boxplot(alt_df=testres, base_df=testres_bas)
plt.show()
#plt.savefig('Boxplot of victory time fitness functions.jpg') #Uncomment and comment plt.show() to save an image of the plot
