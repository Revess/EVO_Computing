# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Variables
top = 160 #160 is all
folderCSV = "./data/CSV/"
fileCSV="generalistsTestRes.csv"
folderOutput = "./plots/"
duplicates_to_drop = "ids"

# Function(s)
def make_plot(df):
    indexes = []
    df.reset_index(inplace=True)
    
    # Create the boxplot
    for ea in [1,2]:
        for g in [1,2]:
            for t in ["S","R"]:
                data = df.loc[(df["EA"] == ea) & (df["Group"] == str(g)+t)].nlargest(top,"Fitness")
                for i, r in data.iterrows():
                                indexes.append(i)

    # Rename and sort values
    df = df.sort_values(by="EA", ascending=True)
    df = df.sort_values(by="Group", ascending=True)
    df['EA'] = df['EA'].replace([1],'Tournament')
    df['EA'] = df['EA'].replace([2],'Roulette')
    df['Group'] = df['Group'].replace(['1R'],'Baseline Group1')
    df['Group'] = df['Group'].replace(['2R'],'Baseline Group2')
    df['Group'] = df['Group'].replace(['1S'],'Specialist Group1')
    df['Group'] = df['Group'].replace(['2S'],'Specialist Group2')
    #df.rename(columns = {'Group': 'Enemy'}, inplace = True)

    print(df)
    hue = "EA"
    sns.boxplot(data = df, x = "Group", y = "Fitness", hue = hue, width = 0.6)
    plt.title("Comparison of baseline and specialists on group 1 and 2".format(k = top))
    fileOutput = folderOutput + "all_specialists_per_enemy_{hue}_boxplot.png".format(top = top, hue = hue)
    plt.savefig(fileOutput, dpi=1000)
    plt.show()
    plt.clf()
    print("Saved the boxplot to: " + str(fileOutput))

# Retrieve the dataframe
trainres = pd.read_csv(folderCSV + fileCSV)

# Create the folder if not exists
if not os.path.exists(folderOutput):
    os.makedirs(folderOutput)

# Make the plot
make_plot(trainres)