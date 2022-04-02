import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import os
import time
import pickle

class NeuralisHalo():

	def __init__(self):
		self.mlp = MLPRegressor(hidden_layer_sizes=(10, 5), # (10, 5)
	                            activation='tanh', # relu, tanh, logistic
	                            solver='adam',
	                            batch_size='auto',
	                            learning_rate_init=0.01,
	                            max_iter=1,                           # incremental learning - one step
	                            shuffle=False,                        # erre is oda kell figyelni
	                            random_state=1,
	                            verbose=True, 
	                            warm_start=True,
	                            momentum=0.9,
	                            nesterovs_momentum=True,
	                            early_stopping=True,
	                            n_iter_no_change=98765000)

		