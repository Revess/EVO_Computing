import pandas as pd
import matplotlib.pyplot as plt
import generalist.src.plotFunctions as pf
import seaborn as sns

# Get gain plot
sns.set(font_scale = 1.2, style = 'ticks')
df_test = pd.read_csv('./data/CSV/testScoresGeneralists.csv')
pf.get_gain_boxplot(df_test)
plt.savefig('./plots/testboxplot.jpg')

# Get subplot of train plots

trainres = pd.read_csv('./data/CSV/trainingGeneralists.csv')
width = 2
height = 2

fig, ax = plt.subplots(height,width)
fig.suptitle('Fitness values over generations per group per type', x =0.5, y = 0.94)

ev = 1
for i in range(height):
    if i == 0:
        type = 'R'
    elif i == 1:
        type = 'S'
    for j in range(width):
        if j == 0:
            group = 1
        elif j == 1:
            group = 2
        pf.get_train_plot(df=trainres, grp=group, EA_number=ev, type = type, ax=ax[i][j])

plt.tight_layout()
fig.legend()
fig.set_size_inches(10,6)
plt.savefig('./plots/trainingplots.jpg')




