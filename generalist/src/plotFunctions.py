import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_train_plot(df,grp,EA_number,type, ax):
    df = df[df.Group == grp]
    df = df[df.EA == EA_number]
    df = df[df.Type == type]

    # Mean of maxs results
    df_max = df.drop(['EA', 'Group', 'Type', 'Individual', 'Player Health', 'Enemy Health', 'Time'], axis = 1)\
        .groupby(['Round', 'Generation'], as_index=False).max()

    mean_max_grouped = df_max.groupby('Generation').mean().drop('Round', axis=1)

    max_grouped_std = df_max.groupby('Generation').std().drop('Round', axis=1)
    max_grouped_std = max_grouped_std.rename(columns={'Fitness': 'max_std'})

    df_max_results = pd.concat([mean_max_grouped, max_grouped_std], axis=1)

    # Mean of means results

    df_mean = df.drop(['EA', 'Group', 'Type', 'Individual', 'Player Health', 'Enemy Health', 'Time'], axis = 1)\
        .groupby(['Round', 'Generation'], as_index=False).mean()

    mean_grouped = df_mean.groupby('Generation').mean().drop('Round', axis=1)
    mean_grouped_std = df_mean.groupby('Generation').std().drop('Round', axis=1)
    mean_grouped_std = mean_grouped_std.rename(columns={'Fitness': 'mean_std'})

    df_mean_results = pd.concat([mean_grouped, mean_grouped_std], axis=1)
    title = 'Group ' + str(grp) + ', type ' + type
    if EA_number ==1 and grp == 1 and type == 'S':
        ax.plot(df_max_results.index, df_max_results.Fitness, 'ro', label='mean of maxs')
    else:
        ax.plot(df_max_results.index, df_max_results.Fitness, 'ro')
    ax.plot(df_max_results.index, df_max_results.Fitness, 'r-')
    ax.fill_between(df_max_results.index, df_max_results.Fitness - df_max_results.max_std,
                df_max_results.Fitness + df_max_results.max_std, color='red', alpha=0.1)

    # Mean plot
    if EA_number == 1 and grp == 1 and type == 'S':
        ax.plot(df_mean_results.index, df_mean_results.Fitness, 'o', label='mean of means')
    else:
        ax.plot(df_mean_results.index, df_mean_results.Fitness, 'o')
    ax.plot(df_mean_results.index, df_mean_results.Fitness, 'b-')
    ax.fill_between(df_mean_results.index, df_mean_results.Fitness - df_mean_results.mean_std,
                df_mean_results.Fitness + df_mean_results.mean_std, color='blue', alpha=0.1)

    ax.set_title(title)
    ax.set_ylabel('Fitness value')
    ax.set_xlabel('Generation')

def get_gain_boxplot(df):
    df_test = df

    df_test['gain_min'] = df_test['Player Health'] - df_test['Enemy Health']
    gain = df_test.drop(['Player Health','Fitness','Enemy Health', 'EA','Generation','Individual','Testing Round','Time','Weights'], axis=1)
    gain = gain.groupby(['Group','Type','Training Round'], as_index= False).sum().drop(['Enemy'], axis = 1)

    gain['EA type'] = 'No type'

    for i in range(gain.gain_min.size):
        gain.loc[i,'EA type'] = str(gain.iloc[i,0]) + gain.iloc[i,1]


    gain_wide = gain.pivot(index = 'Training Round', columns= 'EA type', values = 'gain_min')

    sns.boxplot(gain_wide, width=0.3)
    plt.ylabel('Gain score')
    plt.title('Boxplot of gain score per EA method')

