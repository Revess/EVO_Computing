# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Variables
top = 750
folderCSV = "./data/CSV/"
fileCSV = "trainingSpecialists.csv"
folderOutput = "./plots/"
fileOutput = folderOutput + "top_{top}_specialists_per_enemy_boxplot.png".format(top=top)
duplicates_to_drop = "Fitness"

# Function(s)
def make_plot(df):
    indexes = []
    df.reset_index(inplace=True)

    # Find the fittest members of each EA and Enemy (Prepare the DataFrame)
    for EA_nr in range(1,3):
        for enem in range(1,9):
            perenemy = df.loc[(df["Enemy"] == enem)&(df["EA"] == EA_nr)]
            sorted = perenemy.sort_values(by="Fitness", ascending=True).drop_duplicates(duplicates_to_drop, keep='last')[-top:]
            for i, r in sorted.iterrows():
                indexes.append(i)
    
    # Rename the column EA for the legend
    df['EA'] = df['EA'].replace([1],'Tournament')
    df['EA'] = df['EA'].replace([2],'Roulette')

    # Create the boxplot
    sns.boxplot(data=df.loc[df["index"].isin(indexes)], x="Enemy", y="Fitness", hue="EA", width=0.6)
    plt.title("Comparison of top {k} specialists by enemy".format(k=top))
    plt.savefig(fileOutput, dpi=800)
    print("Saved the boxplot to: " + str(fileOutput))

# Retrieve the dataframe
trainres = pd.read_csv(folderCSV + fileCSV)

# Create the folder if not exists
if not os.path.exists(folderOutput):
    os.makedirs(folderOutput)

# Make the plot
make_plot(trainres)