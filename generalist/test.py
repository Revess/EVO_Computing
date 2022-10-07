import sys, os
sys.path.insert(0, './evoman') 
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pickle as pkl
import pandas as pd
import numpy as np
from deap import creator, base
from environment import Environment
from src.demo_controller import player_controller, enemy_controller
from math import tanh
from tqdm import tqdm

#Initialize
creator.create("Fitness", base.Fitness, weights=(1.0,))      
creator.create("IndividualContainer", list, fitness=creator.Fitness)

##DO SOMETHING SMART HERE FOR TESTING ALL Best AI 5 times