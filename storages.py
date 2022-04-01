import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import os
import pickle

import importlib
import sys

class Storage():
  def __init__(self):
    self.name = 'Storage'
    self.pkl_mlp_filename = 'pickle_mlp.pkl'
    self.pkl_regression_filename = 'pickle_regression.pkl'
    self.pkl_minmaxscaler_filename = 'pickle_minmaxscaler.pkl'

  def save_all_from_object(self, car):
    # Save to file in the current working directory
    # car.mlp
    with open(self.pkl_mlp_filename, 'wb') as file:
      pickle.dump(car.mlp, file)
    # car.regression
    tuple_objects_regression = (car.regression_left, car.regression_center, car.regression_right)
    with open(self.pkl_regression_filename) as file:
      pickle.dump(tuple_objects_regression, file)
    # car.minmaxscaler
    tuple_objects_minmaxscaler = (car.x_minmaxscaler, car.y_minmaxscaler)
    with open(self.pkl_minmaxscaler_filename, 'wb') as file:
      pickle.dump(tuple_objects_minmaxscaler, file)
      

  def save_mlp(self, mlp):
    # Save to file in the current working directory
    pkl_filename = self.pkl_mlp_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(mlp, file)

  def save_mlp_from_object(self, car):
    # Save to file in the current working directory
    pkl_filename = self.pkl_mlp_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(car.mlp, file)

  def load_mlp(self):
    # Load from file
    pkl_filename = self.pkl_mlp_filename
    with open(pkl_filename, 'rb') as file:
      self.mlp = pickle.load(file)
      return self.mlp
  
  def save_regression(self, regression_left, regression_center, regression_right):
    # Save to file in the current working directory
    tuple_objects = (regression_left, regression_center, regression_right)
    pkl_filename = self.pkl_regression_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(tuple_objects, file)

  def save_regression_from_object(self, car):
    # Save to file in the current working directory
    tuple_objects = (car.regression_left, car.regression_center, car.regression_right)
    pkl_filename = self.pkl_regression_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(tuple_objects, file)

  def load_regression(self):
    # Load from file
    pkl_filename = self.pkl_regression_filename
    regression_left, regression_center, regression_right = pickle.load(open(pkl_filename, 'rb'))
    self.regression_left = regression_left
    self.regression_center = regression_center
    self.regression_right = regression_right

  def save_minmaxscaler(self, x_minmaxscaler, y_minmaxscaler):
    # Save to file in the current working directory
    tuple_objects = (x_minmaxscaler, y_minmaxscaler)
    pkl_filename = self.pkl_minmaxscaler_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(tuple_objects, file)

  def save_minmaxscaler_from_object(self, car):
    # Save to file in the current working directory
    tuple_objects = (car.x_minmaxscaler, car.y_minmaxscaler)
    pkl_filename = self.pkl_minmaxscaler_filename
    with open(pkl_filename, 'wb') as file:
      pickle.dump(tuple_objects, file)

  def load_minmaxscaler(self):
    # Load from file
    pkl_filename = self.pkl_minmaxscaler_filename
    x_minmaxscaler, y_minmaxscaler = pickle.load(open(pkl_filename, 'rb'))
    self.x_minmaxscaler = x_minmaxscaler
    self.y_minmaxscaler = y_minmaxscaler




