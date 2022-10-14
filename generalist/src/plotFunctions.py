import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Train results
def get_train_plot(data_frame, group_number, evo_number, ax):
    enemy = data_frame[data_frame.enemy == group_number]
    enemy = enemy[enemy.evo == evo_number]

    # Mean of maxs results
    df_max = enemy.drop(['evo', 'enemy', 'playerHealth', 'enemyHealth', 'time', 'individual'], axis=1).groupby(
        ['run', 'generation'], as_index=False).max()

    max_grouped = df_max.groupby('generation').max().drop('run', axis=1)
    max_grouped_std = df_max.groupby('generation').std().drop('run', axis=1)
    max_grouped_std = max_grouped_std.rename(columns={'fitness': 'max_std'})

    df_max_results = pd.concat([max_grouped, max_grouped_std], axis=1)

    # Mean of means results
    df_mean = enemy.drop(['evo', 'enemy', 'playerHealth', 'enemyHealth', 'time', 'individual'], axis=1).groupby(
        ['run', 'generation'], as_index=False).mean()
    mean_grouped = df_mean.groupby('generation').mean().drop('run', axis=1)
    mean_grouped_std = df_mean.groupby('generation').std().drop('run', axis=1)
    mean_grouped_std = mean_grouped_std.rename(columns={'fitness': 'mean_std'})

    df_mean_results = pd.concat([mean_grouped, mean_grouped_std], axis=1)

    # Max plot
    if evo_number == 0:
        evo_tit = 'Tournament'
    elif evo_number == 1:
        evo_tit = 'Roulette'
    else:
        evo_tit = 'Error: unknown method'
    title = 'Enemy ' + str(enemy_number) + ', ' + evo_tit
    if enemy_number == 2 and evo_number:
        ax.plot(df_max_results.index, df_max_results.fitness, 'ro', label='mean of maxs')
    else:
        ax.plot(df_max_results.index, df_max_results.fitness, 'ro')
    ax.plot(df_max_results.index, df_max_results.fitness, 'r-')
    ax.fill_between(df_max_results.index, df_max_results.fitness - df_max_results.max_std,
                    df_max_results.fitness + df_max_results.max_std, color='red', alpha=0.1)

    # Mean plot
    if enemy_number == 2 and evo_number:
        ax.plot(df_mean_results.index, df_mean_results.fitness, 'o', label='mean of means')
    else:
        ax.plot(df_mean_results.index, df_mean_results.fitness, 'o')
    ax.plot(df_mean_results.index, df_mean_results.fitness, 'b-')
    ax.fill_between(df_mean_results.index, df_mean_results.fitness - df_mean_results.mean_std,
                    df_mean_results.fitness + df_mean_results.mean_std, color='blue', alpha=0.1)

    ax.set_xticks(ticks=range(0, 21, 2))
    ax.set_title(title)
    if enemy_number == 2:
        ax.set_ylabel('Fitness value')
    else:
        ax.set_ylabel('Fitness value')
    if evo_number == 1:
        ax.set_xlabel('Generation')
    else:
        ax.set_xlabel('Generation')

# Test results
#EA 0
def get_test_boxplot(data_frame):
    testres = data_frame
    testres = testres.rename(columns={'round': 'run'})
    test_mean = testres.groupby(['evo','enemy','run'], as_index=False).mean().drop('individual', axis = 1)
    test_mean['gain_min'] = test_mean['p'] - test_mean['e']

    evo = [0] * 10 + [1] * 10
    run = list(range(10)) * 2
    gain_ar = {'ea':evo, 'run':run}
    gain_df = pd.DataFrame(gain_ar)
    gain_df['gain'] = ['nothing'] * gain_df['run'].size

    for e in gain_df['ea']:
        if e == 0:
            ran = range(0,10)
        elif e == 1:
            ran = range(10,20)
        for r in ran:
            gain = 0
            for g in range(test_mean.run.size):
                if gain_df.loc[r,'run'] == test_mean.loc[g,'run'] and e == test_mean.loc[g,'evo']:
                    gain+= test_mean.loc[g,'gain_min']
            gain_df.loc[r,'gain'] = gain

    gain_df['gain'] = gain_df['gain'].astype('float')
    gain_wide = gain_df.pivot(index='run', columns='ea', values='gain')
    gain_wide.columns = ['Tournament','Roulette']
    sns.boxplot(gain_wide, width=0.3)
    plt.title('Boxplot of gain score per EA method')
    plt.xlabel('EA method')
    plt.ylabel('Gain score')


def bas_alt_mutation(alternative, baseline):
    testres_alt = alternative
    testres_alt = testres_alt[testres_alt.p != 0].groupby(['evo', 'enemy', 'individual', 'round'],as_index=False).mean()

    testres_bas = baseline
    testres_bas = testres_bas[testres_bas.p != 0].groupby(['evo', 'enemy', 'individual', 'round'],as_index=False).mean()

    return testres_alt, testres_bas

def get_bas_alt_scatter(alt_df, base_df):
    testres_alt, testres_bas = bas_alt_mutation(alt_df, base_df)
    plt.scatter(testres_alt.t, testres_alt.p, c = 'blue', label = 'Alternative fitness')
    plt.scatter(testres_bas.t,testres_bas.p, c = 'red', label = 'Baseline fitness')
    plt.title('Different fitness functions tested against \n the player health vs the victory time')
    plt.xlabel('Victory time')
    plt.ylabel('Player health left')
    plt.legend()

def get_bas_alt_ph_boxplot(alt_df, base_df):
    testres_alt, testres_bas = bas_alt_mutation(alt_df, base_df)
    player_health = pd.concat([pd.DataFrame(testres_bas.p), pd.DataFrame(testres_alt.p)], axis = 1, keys = ['baseline','alternative'])
    player_health.columns = ['Baseline player health','Alternative player health']

    sns.boxplot(player_health, width=0.3)
    plt.title('Distribution of the player health left')
    plt.ylabel('Player health left')
    plt.xlabel('Fitness function')

def get_bas_alt_vt_boxplot(alt_df, base_df):
    testres_alt, testres_bas = bas_alt_mutation(alt_df, base_df)
    victory_time = pd.concat([pd.DataFrame(testres_bas.t), pd.DataFrame(testres_alt.t)], axis = 1, keys = ['baseline','alternative'])
    victory_time.columns = ['Baseline victory time','Alternative victory time']
    sns.boxplot(victory_time, width=0.3)
    plt.title('Distribution of the victory time')
    plt.ylabel('Victory time')
    plt.xlabel('Fitness function')