Checkpoint name formatting:
EA_Group_popType_Round
Example of EA1 with Group1 populationType Specialist and Round0:
1_1_S_0
Example of EA1 with Group1 populationType Specialist and Round0:
2_1_S_0

The specialsts work slightly differently:
Specialist+Num_Enemy_Round
Example of Specialist 1 with Enemy 1 and Round0:
S1_1_0
Example of Specialist 2 with Enemy 1 and Round0:
S2_1_0

These pickled files can be found in ./data/checkpoints/[EA_Group_popType_Run]/[GENERATION].pkl

This formatting also goes for the final trained population and for the final results that can be found in:
"./data/trainedPopulations/[EA_Group_popType_Run].pkl"
"./data/trainedPopulations/[Specialist+Num_Enemy_Round].pkl"
Here the data is formatted that each list inside the trained populations contains the entire population of the generation, index 0 it is initial generation

"./data/trainedPopulations/[EA_Group_popType_Run].pkl"
"./data/trainingresults/[Specialist+Num_Enemy_Round].pkl"
Here the data is formatted by:
    -Fitness (list where every entry is a generation and each index inside is an individual's score)
    -Player Health (list where every entry is a generation and each index inside is an individual's score)
    -Enemy Health (list where every entry is a generation and each index inside is an individual's score)
    -Time (list where every entry is a generation and each index inside is an individual's score)
    -The entire history Class of Deap
    -The entire Hall of Fame class of Deap

To train the EA please carefully edit the settings (A restore file can be found in ./bak):
    -Here the first nest gives the type of EA
    -The second gives the Settings in the case that you specified EA1 or EA2:
        -Then there will be the type of EA and Group you want to train (S_1) = Specialist base group 1, (R_2) is random base group 2
        -Then in there you can set the settings for that specific EA/Group etc.
    -The second argument for the Specialist is similar to the settings of EA1 and EA2 but dont have a group or type specifier, also the specialist must always train all enemies!!
    -The noise argument in the settings must be a float between 0 and 1 to specify the percentage of population that should be noise
    
python train.py -h
This will run the train.py file, read the help file to know what you are doing

python test.py -h
Will test bset of your trained population

genCSV.py will generate a .csv file for all data, to be found in ./data/CSV/