import pandas as pd
import matplotlib.pyplot as plt

trainres = pd.read_csv('trainingResults.csv')

def get_plot(data_frame, enemy_number, evo_number, ax):

    enemy = data_frame[data_frame.enemy == enemy_number]
    enemy = enemy[enemy.evo == evo_number]

    # Mean of maxs results
    df_max = enemy.drop(['evo','enemy','playerHealth','enemyHealth','time','individual'],axis = 1).groupby(['run','generation'], as_index= False).max()
    
    max_grouped = df_max.groupby('generation').max().drop('run', axis = 1)
    max_grouped_std = df_max.groupby('generation').std().drop('run', axis = 1)
    max_grouped_std = max_grouped_std.rename(columns={'fitness':'max_std'})

    df_max_results = pd.concat([max_grouped, max_grouped_std], axis = 1)

    # Mean of means results
    df_mean = enemy.drop(['evo','enemy','playerHealth','enemyHealth','time','individual'],axis = 1).groupby(['run','generation'], as_index= False).mean()
    mean_grouped = df_mean.groupby('generation').mean().drop('run', axis = 1)
    mean_grouped_std = df_mean.groupby('generation').std().drop('run', axis = 1)
    mean_grouped_std = mean_grouped_std.rename(columns={'fitness':'mean_std'})

    df_mean_results = pd.concat([mean_grouped, mean_grouped_std], axis = 1)

    # Max plot
    title = 'enemy = '+str(enemy_number) + ', EA-method =' + str(evo_number)
    if enemy_number == 2 and evo_number:
        ax.plot(df_max_results.index,df_max_results.fitness, 'ro', label = 'mean of maxs')
    else:
        ax.plot(df_max_results.index, df_max_results.fitness, 'ro')
    ax.plot(df_max_results.index,df_max_results.fitness, 'r-')
    ax.fill_between(df_max_results.index, df_max_results.fitness - df_max_results.max_std, df_max_results.fitness + df_max_results.max_std,color='red', alpha=0.1)

    # Mean plot
    if enemy_number == 2 and evo_number:
        ax.plot(df_mean_results.index,df_mean_results.fitness, 'o', label = 'mean of means')
    else:
        ax.plot(df_mean_results.index, df_mean_results.fitness, 'o')
    ax.plot(df_mean_results.index,df_mean_results.fitness, 'b-')
    ax.fill_between(df_mean_results.index, df_mean_results.fitness - df_mean_results.mean_std, df_mean_results.fitness + df_mean_results.mean_std,color='blue', alpha=0.1)

    ax.set_xticks(ticks = range(0,21,2))
    ax.set_title(title)
    if enemy_number == 8:
        ax.set_xlabel('Generation')
    else:
        ax.set_xticks([], [])
    if evo_number == 0:
        ax.set_ylabel('Fitness value')
    else:
        ax.set_yticks([], [])


# Save plot
fig, ax = plt.subplots(3,2)
fig.suptitle('Fitness values over generations per enemy per EA-method', x =0.5, y = 0.92)
for i in range(3):
    if i == 0:
        en = 2
    elif i == 1:
        en = 5
    elif i == 2:
        en = 8
    for j in range(2):
        if j == 0:
            ev = 0
        elif j == 1:
            ev = 1

        get_plot(data_frame=trainres, enemy_number=en,evo_number=ev, ax=ax[i][j])


plt.tight_layout()
fig.legend(loc="upper right")
fig.set_size_inches(6,8)

plt.savefig('Fitness over gen per enemy per EA.jpg')


