from plotting import plot
import pickle as pkl
import os
from datetime import datetime
import pandas as pd

f,p,e,t = [None]*4
with open("./finetuning/Run2.pkl", "rb") as file_:
    f,p,e,t = pkl.load(file_)

folderpath = "./plots/"
filepath = folderpath + datetime.now().strftime("%d%m%y_%H%M") + '_results_log.xlsx'

if not os.path.exists(folderpath):
    os.mkdir(folderpath)

fitness_df = pd.DataFrame(f).transpose()
fitness_df['individual'] = fitness_df.index
fitness_df = pd.melt(fitness_df, id_vars='individual', value_vars=range(0, len(f)))
fitness_df = fitness_df.sort_values(['variable','individual'])
fitness_df = fitness_df.reset_index(drop=True)
fitness_df = fitness_df.rename(columns={'variable':'generation', 'value':'fitness value'})

#fitness_df.to_excel(filepath)

plot()