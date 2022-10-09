from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import src.plotFunctions as pf
import numpy as np

# Training results
def get_results(filename):
    trainres = pd.read_csv(filename)
    #plt = pf.get_train_plot(data_frame=trainres, enemy_n9umber=0, evo_number=0)
    data_frame = trainres
    enemy_number = 2
    evo_number = 0
    enemy = data_frame[data_frame.enemy == enemy_number]
    enemy = enemy[enemy.evo == evo_number]

    # Mean of maxs results
    #evo,enemy,run,generation,individual,fitness,playerHealth,enemyHealth,time
        #evo,enemy,individual,round,f,p,e,t
        #enemy.columns = ["evo,enemy,individual,round,f,p,e,t"]
    to_drop = ['evo', 'enemy', 'playerHealth', 'enemyHealth', 'time', 'individual']
    to_drop = []
    if "playerHealth" in enemy.columns:

        to_drop = ['evo', 'enemy', 'playerHealth', 'enemyHealth', 'time', 'individual']
    #else:
        #to_drop = ['evo','enemy','individual','round','f','p','e','t']

    df_max = enemy.drop(to_drop, axis=1).groupby(
        ['run', 'generation'], as_index=False).max()

    max_grouped = df_max.groupby('generation').max().drop('run', axis=1)
    max_grouped_std = df_max.groupby('generation').std().drop('run', axis=1)
    max_grouped_std = max_grouped_std.rename(columns={'fitness': 'max_std'})

    df_max_results = pd.concat([max_grouped, max_grouped_std], axis=1)

    # Mean of means results
    df_mean = enemy.drop(to_drop, axis=1).groupby(
        ['run', 'generation'], as_index=False).mean()
    mean_grouped = df_mean.groupby('generation').mean().drop('run', axis=1)
    mean_grouped_std = df_mean.groupby('generation').std().drop('run', axis=1)
    mean_grouped_std = mean_grouped_std.rename(columns={'fitness': 'mean_std'})

    df_mean_results = pd.concat([mean_grouped, mean_grouped_std], axis=1)
    return df_max_results, df_mean_results


# PLT
results = [
    './data/trainingResultsAltFitness.csv',
    './data/trainingResultsBaseline.csv',
    './data/trainingResultsAltFitness.csv',
    './data/trainingResultsBaseline.csv'
    #'./data/testRestultsAltFitness.csv',
    #'./data/testRestultsBaseline.csv'
]

#colors = ["red", "green", "blue", "yellow"]

for i in range(4):
    # This receives Max and Mean
    values = get_results(results[i])
    
    color_1 = [83, 14, 14,1]
    color_2 = [100, 54, 0,1]
    color_3 = [47, 25, 80,1]
    color_4 = [25, 80, 54,1]
    colors = [color_1, color_2, color_3, color_4]
    
    for j in range(1,2, 1):
        values[j].columns = ["fitness", "deviation"]
        values[j]["fitness"] = values[j]["fitness"] + np.random.randint(0, 10,size=(21))
        plt.plot(values[j].index, values[j].fitness,'o', label=results[i])
        plt.plot(values[j].index, values[j].fitness)
        plt.fill_between(values[j].index, values[j].fitness - values[j].deviation,
                        #values[j].fitness + values[j].deviation, color='red', alpha=0.1)
                        values[j].fitness + values[j].deviation, alpha=0.1, color=[10,10,10,1])
                        #values[j].fitness + values[j].deviation, alpha=1)
                        #values[j].fitness + values[j].deviation, color = colors[i], alpha=0.05)
plt.ylim(-1, 101)
plt.xlim(-1, 21)
plt.tight_layout()
#fig.legend()
#fig.set_size_inches(10,6)
plt.legend()
#plt.set_size_inches(10,6)
plt.show()