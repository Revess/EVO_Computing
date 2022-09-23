import pandas as pd
import matplotlib.pyplot as plt
import os

# Loading most recently generated dataframe
folder = 'log/'
dirpath = os.path.dirname(__file__)
filepath = os.path.join(dirpath, folder)
entries = os.listdir(filepath)
entries.sort()
filename = filepath + entries[-1]

# Change filename to another path for a specific dataset to open
data = pd.read_excel(filename, index_col = 0)

# Make fitness mean,max,min per gen plot
df_max = data.groupby('generation', as_index= False).max().drop('individual', axis = 1)
df_max = df_max.rename(columns={'fitness value':'max'})
df_mean = data.groupby('generation', as_index= False).mean().drop('individual', axis = 1)
df_mean = df_mean.rename(columns={'fitness value':'mean'})
df_min = data.groupby('generation', as_index= False).min().drop('individual', axis = 1)
df_min = df_min.rename(columns={'fitness value':'min'})

df_results = pd.concat([df_max,df_mean,df_min], axis = 1).drop('generation', axis = 1)

ax = df_results.plot(title = 'Fitness over generations')
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
plt.show()

# Make fitness boxplot per gen
# Using many generations ruins this plot. Can be used but it is not clear with many generations

wide = data.pivot(index='individual', columns='generation', values='fitness value')
ax1 = wide.boxplot()#title = 'Distribution of the fitness over generations')
ax1.set_xlabel('Generation')
ax1.set_ylabel('Fitness value')

plt.show()

# Make heatmap
fig, ax = plt.subplots()
im = ax.imshow(wide, interpolation=)
plt.show()