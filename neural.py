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
                                validation_fraction=0.0,
                                verbose=False, 
                                warm_start=True,
                                momentum=0.9,
                                nesterovs_momentum=True,
                                early_stopping=False,
                                n_iter_no_change=98765000)

        self.x_minmaxscaler = MinMaxScaler(feature_range=(-1,1))       # A range amibe skálázunk (-1, 1)
        self.y_minmaxscaler = MinMaxScaler(feature_range=(-1,1))       # A range amibe skálázunk (-1, 1)

        self.mlp_fit_evaluation_time_holder = []
        self.predicted_last_holder      = []

    def scale_inputs(self, X_inputs):
        self.x_minmaxscaler.fit(X_inputs)
        self.X_scaled = self.x_minmaxscaler.transform(X_inputs)

    def scale_output(self, y_target):
        self.y_minmaxscaler.fit(y_target)
        self.y_scaled = self.y_minmaxscaler.transform(y_target) 

    def print_hello(self):
        print('hello')

    def fit(self):
        mlp_fit_time_start = time.time()
        self.mlp.fit(self.X_scaled, self.y_scaled)
        mlp_fit_time = time.time() - mlp_fit_time_start
        self.mlp_fit_evaluation_time_holder.append(mlp_fit_time)

    def predict_for_all(self):
        self.predicted_all = self.mlp.predict(self.X_scaled)

    def predict_for_last(self):
        self.predicted_last = self.mlp.predict(self.X_scaled[-1:,])
        self.predicted_last_holder.append(self.predicted_last)




