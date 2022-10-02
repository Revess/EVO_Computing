Hi and welcome to the repository of Group 52.

This repository is currently serving the function of the training, testing and evaluating EA for assignment 1. 
This means that only the specialist functions will be run, no generalist EA will be trained here.

The repository consists of a couple of key features so everything can be run smoothly:

Start of by running ```pip install -r requirements.txt``` so your python will be up-to-date to run any of the 
python scripts we prepared

Secondly there is a settings.txt file, here you can find any settings that the EA is trained on and can be edited to train the EA
differently.
The settings are:
    -neurons: 100           -Set the number of neurons, INT
    -geneMin: -1            -NOT USED ANYMORE
    -geneMax: 1             -NOT USED ANYMORE
    -popsize: 100           -Set the population size for during training, INT
    -cxpb: 0.5              -Set the crossover probability, FLOAT
    -mutpb: 0.3             -Set the mutation probability, FLOAT
    -ngens: 20              -Set the number of generations, INT
    -npools: 10             -NOT USED ANYMORE
    -mu: 0                  -Set the MU of the mutation function, FLOAT
    -sigma: 1               -Set the sigma of the mutation function, FLOAT
    -indpb: 0.1             -Set the probability step op the mutation function, FLOAT
    -tournsize: 3           -Set the tournament size (will be ignored when tournament is set to False), INT
    -name: 0_2_0            -Set the name of the single run (will be overwritten when running any other file), STR
    -finetuning: True       -NOT USED ANYMORE
    -tournament: True       -Set wheather to use tournament selection (if False it will use roulette), BOOL
    -enemy: 2               -Set which enemy to train on (will be overwritten when running any other file), INT (1-8)
Failing to set the settings properly will crash the script

With singlerun.py you can train a single EA with the settings in settings.txt .
When you want to complete a full training run use trainSpecialists.py . This will run all the enemies and EA 10 times.
Finishing trainSpecialists.py the results can be found in dataBackup, then you can test the EA with testData.py, this is used to
pick the best individuals and run them again. Make sure to edit the file to select the correct .csv files.
And finally to generate some fun plots run generateResults.py

Inside ./src there are some backend files and a statistical test notebook to see similarity between the baseline fitness and our custom 
function.
Finally there is the ./data folder, here the populations and fitnessvalues will be dumped. There you can also find our results for our four trained EA, the AltFitness files and folders are the custom fitness EA and the baseline are the given fitness EA.

For more workflow items etc see our GIT: https://github.com/Revess/EVO_Computing