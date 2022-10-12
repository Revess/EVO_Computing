# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Variables
top = 160
folderCSV = "./data/CSV/"
#fileCSV = "trainingSpecialists.csv"
#fileCSV = "trainingGeneralists.csv"
#fileCSV = "niks.csv"
#fileCSV ="specialistsTestRes.csv"
fileCSV="generalistsTestRes.csv"
folderOutput = "./plots/"
duplicates_to_drop = "ids"

# Function(s)
def make_plot(df):
    indexes = []
    df.reset_index(inplace=True)
    #print(df)
    '''
    # Find the fittest members of each EA and Enemy (Prepare the DataFrame)
    for EA_nr in range(1,3):
        for group in range(1,3):
            for type in ["R", "S"]:
                print("EA_nr: {EA_nr} - group: {group} - type: {type}".format(EA_nr=EA_nr, group = group, type=type))
                perenemy = df.loc[(df["Group"] == group)&(df["EA"] == EA_nr)&(df["Type"] == type)]
                sorted = perenemy.sort_values(by="Fitness", ascending=True)#.drop_duplicates(duplicates_to_drop, keep='last')[-top:]
                
                print("Enemy: {en} - EA_nr: {ea} - unique Fitness out of {top}: {uf}".format(en=enem, ea=EA_nr,top=top, uf=sorted["Fitness"].unique().shape[0]))
                print("Enemy: {en} - EA_nr: {ea} - unique Player Health out of {top}: {uf}".format(en=enem, ea=EA_nr,top=top, uf=sorted["Player Health"].unique().shape[0]))
                print("Enemy: {en} - EA_nr: {ea} - unique Time out of {top}: {uf}".format(en=enem, ea=EA_nr,top=top, uf=sorted["Time"].unique().shape[0]))
                print("Enemy: {en} - EA_nr: {ea} - unique Enemy Health out of {top}: {uf}".format(en=enem, ea=EA_nr,top=top, uf=sorted["Enemy Health"].unique().shape[0]))
                print("Enemy: {en} - EA_nr: {ea} - unique ids out of {top}: {uf}".format(en=enem, ea=EA_nr,top=top, uf=sorted["ids"].unique().shape[0]))

                #print(sorted) 
                print(sorted["Player Health"].unique())
                print()
           
                for i, r in sorted.iterrows():
                    indexes.append(i)

    # Rename the column EA for the legend
    df['EA'] = df['EA'].replace([1],'Tournament')
    df['EA'] = df['EA'].replace([2],'Roulette')
    df['Type'] = df['Type'].replace(['R'],'Baseline')
    df['Type'] = df['Type'].replace(['S'],'Specialist')

    #df.to_csv("./data/CSV/output.csv", index=False)
    '''
    # Create the boxplot
    for ea in [1,2]:
        for g in [1,2]:
            for t in ["S","R"]:
                data = df.loc[(df["EA"] == ea) & (df["Group"] == str(g)+t)].nlargest(top,"Fitness")
                for i, r in data.iterrows():
                                indexes.append(i)

    print(df)
    df = df.sort_values(by="EA", ascending=True)
    df = df.sort_values(by="Group", ascending=True)
    df['EA'] = df['EA'].replace([1],'Tournament')
    df['EA'] = df['EA'].replace([2],'Roulette')
    
    df['Group'] = df['Group'].replace(['1R'],'Baseline 1')
    df['Group'] = df['Group'].replace(['2R'],'Baseline 2')
    df['Group'] = df['Group'].replace(['1S'],'Specialist 1')
    df['Group'] = df['Group'].replace(['2S'],'Specialist 2')

    df.rename(columns={'Group': 'Enemy'}, inplace=True)
    
    #print(df)
    hues = ["EA"]
    for hue in hues:
        print("Hue: {hue}".format(hue=hue))
        sns.boxplot(data=df, x="Enemy", y="Fitness", hue=hue, width=0.6)
        #sns.boxplot(data=df.loc[df["index"].isin(indexes)], x="Group", y="Fitness", hue=hue, width=0.6)
        #sns.boxplot(data=df.loc[df["index"].isin(indexes)], x="Group", y="Fitness", hue="Type", width=0.6)
        #plt.title("Comparison of top {k} generalists by enemy".format(k=top))
        #fileOutput = folderOutput + "top_{top}_specialists_per_enemy_{hue}_boxplot.png".format(top=top, hue=hue)
        #plt.title("Comparison of {k} generalists by enemy".format(k=top))
        plt.title("Comparison of specialists by enemy".format(k=top))
        #fileOutput = folderOutput + "{top}_specialists_per_enemy_{hue}_boxplot.png".format(top=top, hue=hue)
        fileOutput = folderOutput + "all_specialists_per_enemy_{hue}_boxplot.png".format(top=top, hue=hue)
        plt.savefig(fileOutput, dpi=800)
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
#print(trainres)