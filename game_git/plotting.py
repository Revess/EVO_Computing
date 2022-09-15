import pandas as pd
import matplotlib.pyplot as plt
import os

# Loading most recently generated dataframe
folder = 'data/log/'
dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, folder)
entries = os.listdir(filepath)
entries.sort()
filename = filepath + entries[-1]

data = pd.read_excel(filename, index_col = 0)

df_max = data.groupby('generation', as_index= False).max().drop('individual', axis = 1)
df_max = df_max.rename(columns={'value':'max'})
df_mean = data.groupby('generation', as_index= False).mean().drop('individual', axis = 1)
df_mean = df_mean.rename(columns={'value':'mean'})
df_min = data.groupby('generation', as_index= False).min().drop('individual', axis = 1)
df_min = df_min.rename(columns={'value':'min'})

df_results = pd.concat([df_max,df_mean,df_min], axis = 1).drop('generation', axis = 1)

ax = df_results.plot(title = 'Fitness over time')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
plt.show()