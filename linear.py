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

class LinRegression():

	def __init__(self):
		self.regression_metrika1 = LinearRegression(fit_intercept=True)
		self.regression_metrika2 = LinearRegression(fit_intercept=True)
