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
import multiprocessing

from debils import Printer
from environments import Road
from plotters import PostPlotter, Plotter
from storages import Storage

from IPython.display import clear_output

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class TestCar():
  def __init__(self, road, plotter, storage, printer):
    self.plot_frequency = 9
    self.plot_detailed_frequency = 32
    self.plot_history_flag = 0                          # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_trace_flag = 0                            # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plotter_mlp_flag = 0                           # 0 - disable, 1 - plot, 2 - save, 3 - both

    self.road = road
    self.plotter = plotter
    self.storage = storage

    self.sensor_center_enable = True

    # self.action_zero_is_allowed = False               # erre tuliképp a TestCar osztályban nincs is szükség

    self.linear_regression_calculation = 'old'

    self.x = 0
    self.y = self.road.wall_center[0]
    self.sight = 400           # ennyit lát előre 300, 54, 154
    self.sight_center = 400    # ennyit lát előre 150

    self.y_history  = []
    self.x_history  = []
    self.y_center   = self.road.wall_center
    self.y_distance = []
    self.y_distance_real = []
    self.y_distance_predicted = []
    self.y_distance_predicted_inv = []

    self.regression = LinearRegression(fit_intercept=False)

    # Load MLP from file
    self.storage.load_mlp()
    self.mlp = self.storage.mlp

    # Load Regression from file
    self.storage.load_regression
    self.regression_left = self.storage.regression_left
    self.regression_center = self.storage.regression_center
    self.regression_right = self.storage.regression_right

    # Load MinMaxScaler from file
    self.storage.load_minmaxscaler()
    self.x_minmaxscaler = self.storage.x_minmaxscaler
    self.y_minmaxscaler = self.storage.y_minmaxscaler

    # data holders
    self.sensor_center = []
    self.sensor_left   = []
    self.sensor_right  = []
    self.before  = []
    self.after   = []
    
    # error holder
    self.loss_holder = []
    
    # mlp_fit_evaluation_time_holder
    self.mlp_fit_evaluation_time_holder = []

    self.mesterseges_coutner = 0

    # logger helyett
    self.printer = printer
    self.printer._ac = False


  def calculate_distances(self):

    k = self.x; d = 0
    while(k < self.x + self.sight_center):
      k += 1; d += 1
      self.distance_center_from_wall = d
      # v.24 - new
      if( self.sensor_center_enable == True ):
        if(int(self.road.wall_left[k]) < self.y):
          self.printer.sr('Sensor center = ', self.distance_center_from_wall)
          break
        if(int(self.road.wall_right[k]) > self.y):
          self.printer.sr('Sensor center = ', self.distance_center_from_wall)
          break
      if( self.sensor_center_enable == False ):
        self.distance_center_from_wall = 0
        
    # Kiiktatom a középső szenzort -> egyszerűen beállítom az értékét 0-ra
    # self.distance_center_from_wall = 0
            
            
#    k = self.x; d = 0
#    while(k < self.x + self.sight):
#      k += 1;  d += 1
#      self.distance_left_from_wall = d
#      if(int(self.road.wall_left[k]) < self.y + d):
#        self.printer.sr('Sensor from left wall = ', self.distance_left_from_wall)
#        break

    # Ehelyett most az van hogy nézzen simán oldalra
    self.distance_left_from_wall = self.y - self.road.wall_left[self.x]

#    k = self.x; d = 0
#    while(k < self.x + self.sight):
#      k += 1; d += 1
#      self.distance_right_from_wall = d
#      if(int(self.road.wall_right[k]) > self.y - d):
#        self.printer.sr('Sensor from right wall = ', self.distance_right_from_wall)
#        break

    # Ehelyett most az van hogy nézzen simán oldalra
    self.distance_right_from_wall = self.road.wall_right[self.x] - self.y

# ToDo - Ez egy potenciális hiba
    # az ördög soha nem alszik
    self.distance_from_top     = abs(self.road.wall_left[self.x] - self.y)
    self.distance_from_bottom  = abs(self.road.wall_right[self.x] - self.y)
    self.printer.sr('most távolsagra van a felső faltól = ', self.distance_from_top)
    self.printer.sr('most távolsagra van az alsó faltól = ', self.distance_from_bottom)

    # ezt az értéket fogom becsülni, a középértéktől való eltérés mértéke, ha pozitív akkor fölfelé, ha negatív akkor lefelé tér el
    self.vertical_distance_from_middle = self.y - self.road.wall_center[self.x]




  def append(self):
    self.y_distance.append(self.vertical_distance_from_middle)

    self.sensor_left.append(self.distance_left_from_wall)
    self.sensor_center.append(self.distance_center_from_wall)
    self.sensor_right.append(self.distance_right_from_wall)


  def plot_history(self, flag, autoscale = True):
    if( flag != 0 ):
      plot_history(auto = self, flag = flag, autoscale = autoscale)

      # print(' --------------- plot --------------- ')


  def plot_history_fixed(self, flag, ymin, ymax, width, height):
    if( flag != 0 ):
      plot_history_fixed(self, flag, ymin, ymax, width, height)
    
      # print(' --------------- plot --------------- ')


  def plot_history_range(self, flag, start = 0, end = 9999999, autoscale = True):
    if( flag != 0 ):
      plot_history_range(auto = self, flag = flag, start = start, end = end, autoscale = autoscale)

      # print(' --------------- plot --------------- ')


  def save_plots(self):
    plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted)
    plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real');
    plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
    plt.savefig('test_y_distance_vs_y_distance_predicted_{0:04}'.format(self.x)+'.png')
    plt.close()



  def cond1(self, x):
    if (x % 3 == 1):
      return True
    else:
      return False

  def cond2(self, x):
    if (x > -1):
      return True
    else:
      return False

  def cond3(self, x):
    if(x < 500):
      return True
    else:
      return False

  def cond4(self, x):
    # Bármikor lefut ha bármely szenzor értéke kisebb mint
    if( self.sensor_right[-1] < 10 or self.sensor_left[-1] < 10 or self.sensor_center[-1] < 10 ):
      return True
    else:
      return False

  def cond5(self, x):
    # Bármikor lefut ha bármely szenzor értéke kisebb mint
    if( self.y_distance[-1] > 10 or self.y_distance[-1] < -10):
      return True
    else:
      return False

  def run(self, run_length, cond = 1):
    # Mi alapján legyen végrehajtva a tényleges action
    if cond == 1:
      condition_for_action = self.cond1
    if cond == 2:
      condition_for_action = self.cond2
    if cond == 3:
      condition_for_action = self.cond3
    if cond == 4:
      condition_for_action = self.cond4
    if cond == 5:
      condition_for_action = self.cond5


    for i in range(0, run_length, 1):
      self.printer.util('# A run ciklus eleje --------------------------------------------------------------------------------------------------------------------')
      self.printer.util('# i = ', i)
      _summary_mlp_prediction_was_taken = 0
      _summary_mlp_fit_was_taken = 0
      _summary_mesterseges_mozgatas = 0
      _summary_action_was_taken = 0

      self.x = i
      self.calculate_distances()
      self.append()




      # Itt kezdődik a lényeg
      if ( i >= 0 ):

        # A helyzet az, hogy a TestCar nem tanul, ezért erre a részre nem lesz szükségünk
        # --------------------------------------- A NEURÁLIS HÁLÓ TANÍTÁSA (1) ---------------------------------------
        

        # Élhetünk ezzel a lehetőséggel, bár lehet, hogy a végén kiveszünk mindent és csak a döntés marad majd bent
        # --------------------------------------- A NEURÁLIS HÁLÓ MINŐSÉGÉNEK VISSZAMÉRÉSE, TESZTELÉSE (2) ---------------------------------------


        # Igazából ezt a mesterséges mozgatást is kiveszem
        # Tulajdonképpen első körben ezt most benne hagyhatom, de alapvetően majd ki kéne venni
        # ----------------------------------------- MESTERSÉGES MOZGATÁS (3) -----------------------------------------



        # Itt jön a lényeg
        # ------------------------------------------------ ACTION (X) ------------------------------------------------

        action = 0

        # if( i % 3 == 0 ):
        if( i > -1 ):

          if( len(self.before) > 9 ):

            self.printer.info('------------------------------ IF len(self.before) > 9 ------------------------------')
            self.printer.info('\n')


            # most ki kell számolni, hogy mennyi lenne a szenzorok értéke, ha fel le lépkednénk

            move = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])


            self.printer.action('\t # Az egyes lépések várható kimeneteinek kiszámolása ----------------------------------------------')

            self.printer.action('\t\t # Ennyivel mozdulna el egy szenzor adat 1 egység változással ha 1 lenne a before értéke')

            
            action = 0; tmp = 999999990

            for j in move:

              # new 'the original article' linreg calculation
              __right  = self.distance_right_from_wall   # before_array[:,3]
              __center = self.distance_center_from_wall  # before_array[:,2]
              __left   = self.distance_left_from_wall    # before_array[:,1]
              __y = self.y                               # before_array[:,0]
              __j = j                                    #  delta_array[:,0]
              __egy_right    = np.array([__right * __y / (__y + __j)])
              __ketto_right  = np.array([__right * __j / (__y + __j)])
              __X_right_new  = np.stack((__egy_right.flatten(), __ketto_right.flatten()), axis=1)

              __egy_center   = np.array([__center * __y / (__y + __j)])
              __ketto_center = np.array([__center * __j / (__y + __j)])
              __X_center_new = np.stack((__egy_center.flatten(), __ketto_center.flatten()), axis=1)

              __egy_left     = np.array([__left * __y / (__y + __j)])
              __ketto_left   = np.array([__left * __j / (__y + __j)])
              __X_left_new   = np.stack((__egy_left.flatten(), __ketto_left.flatten()), axis=1)

              # old 'pistike fele egyszerű' linreg calculation
              _X_left   = np.array([[self.distance_left_from_wall, j]])
              _X_center = np.array([[self.distance_center_from_wall, j]])
              _X_right  = np.array([[self.distance_right_from_wall, j]])

              if( self.linear_regression_calculation == 'old' ):
                predicted_left   = self.regression_left.predict(_X_left)
                predicted_center = self.regression_center.predict(_X_center)
                predicted_right  = self.regression_right.predict(_X_right)

              if( self.linear_regression_calculation == 'new' ):
                predicted_left   = self.regression_left.predict(__X_left_new)
                predicted_center = self.regression_center.predict(__X_center_new)
                predicted_right  = self.regression_right.predict(__X_right_new)

              # nekünk majd azt az értéket kell választanunk amelyik segítségével a legközelebb jutunk a 0 értékhez

              _X = np.array([predicted_left.ravel(), predicted_center.ravel(), predicted_right.ravel()]).T    # figyelni kell rá, hogy eredetileg is ez volt-e a változók sorrendje

              _X_scaled = self.x_minmaxscaler.transform(_X)

              # Neurális háló becslése
              predicted_position_scaled = self.mlp.predict(_X_scaled)

              # Vissza kell transzformálnom eredeti formájába
              predicted_position = self.y_minmaxscaler.inverse_transform(predicted_position_scaled.reshape(-1, 1))

              # legyünk bátrak és módosítsuk az autó self.y pozicióját

              # azzal az értékkel amely abszolút értékben a legkissebb, helyett
              # mivel a célváltozónk akkor jó ha 0, mivel a középvonaltól mért eltérés
              # ezért itt azt az értéket kell kiválasztani ami a legközelebb van 0-hoz

              # természetesen ezen változtatni kell ha nem a középvonaltól való eltérés mértékét akarjuk becsülni
              # de ahhoz fent is át kell állítani hogy mi legyen a self.y_distance számítása

              if( abs(0 - predicted_position) < tmp):       # rossz volt - javítva - tesztelés alatt
                action = j
                tmp = abs(0 - predicted_position)
                self.printer.action('\t\t ---------------------')
                self.printer.action('\t\t  action = ', action)
                self.printer.action('\t\t  predicted_position = ', predicted_position)
                self.printer.action('\t\t  absolute distance from 0 (tmp) = ', tmp)
                self.printer.action('\t\t ---------------------')

              self.printer.action('\t\t adott j-re {0} kiszámoltuk az előrejelzést de még nem hoztunk döntést -----------------------------------------------------------------'.format(j))
              self.printer.action('\t\t --------------------------------------------------------------------------------------------------------------------------------------')
            
            self.printer.action('\t minden j-re kiszámoltuk az előrejelzést de még nem hoztunk döntést -------------------------------------------------\n')



# A Car osztály korábbi verzióiban a következők szerint volt kiértékelve, hogy mikor hozhat döntést
# version 20. if( i % 3 == 0 ) -> version 22. if( i % 3 == 1 )
# eddig csak akkor engedtem neki lépést, ha ( i % 3 == 0 ):
# most viszont mindíg

# a döntés azonban csak akkor fut le ha az alábbi feltétel teljesül
# Na most egy olyat is bele tudok írni, hogy csak akkor lépjen ha (mondjuk túl közel van a falhoz)

#          if( i % 3 == 0 ):
#          if( i > -1 ):
#          if( self.sensor_right[-1] < 10 or self.sensor_left[-1] < 10 or self.sensor_center[-1] < 10 ):

          take_action = condition_for_action(self.x)

          if( take_action == True ):
            
            # Ha ténylegesen meglépi a döntés csak akkor kerül bele a self.before és a self.after adatba a változók értéke

            self.printer.takeaction('------------------------------ IF i % 3 > 0 ------------------------------')
            _summary_action_was_taken = 1
            self.printer.takeaction('=================== TAKE ACTION ===================')
            self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.printer.takeaction('-------- ennyivel módosítom self.y értékét --------')
            self.printer.takeaction('self.y régi értéke = ', self.y)
            self.y = self.y + action
            self.calculate_distances()
            self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.printer.takeaction('self.y új értéke   = ', self.y)
            self.printer.takeaction('self.y_distance[-1]= ', self.y_distance[-1])
            self.printer.takeaction('action             = ', action)
            self.printer.takeaction('----------------- módosítás vége -----------------')

      # y értékét mindíg hozzá adjuk a self.y_history listához
      self.y_history.append(self.y)
      # adjuk hozzá az értéket a self.y_history-hoz
      self.printer.util('# A run ciklus vége ------------------------------------------------------------------------------------------------------------------------------------------')
      self.printer.util('#   itt adom hozzás a self.y a self.y_history-hoz')
      self.printer.util('#    self.y :')
      self.printer.util(self.y)
      self.printer.util('# \t\t\t --------------- Summary ---------------')
      self.printer.util('# \t\t\t _summary_mlp_fit_was_taken         = ', _summary_mlp_fit_was_taken)
      self.printer.util('# \t\t\t _summary_mlp_prediction_was_taken  = ', _summary_mlp_prediction_was_taken)
      self.printer.util('# \t\t\t _summary_mesterseges_mozgatas      = ', _summary_mesterseges_mozgatas)
      self.printer.util('# \t\t\t _summary_action_were_taken         = ', _summary_action_was_taken)
      self.printer.util('# ')
      self.printer.util('# A run ciklus vége ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

      # Egy nagyon hasznos kiegészítés ha a programot Jupyter Notebookban futtatom
      if ( i % 10 == 0 ):
        clear_output(wait=True)
      
    print('---------- Finish Run ----------')

# A TestCar osztály vége
# ------------------------------------------------------------------------------------------------------------------------------------






































# ------------------------------------------------------------------------------------------------------------------------------------


class Car():
  def __init__(self, road, plotter, storage, printer):
    self.plot_frequency = 9
    self.plot_detailed_frequency = 32
    self.plot_history_flag = 0                          # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_investigation_flag = 0                    # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_before_after_sensor_values_flag = 0       # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_before_after_sensor_estimation_flag = 0   # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_state_space_discover_flag = 0             # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plot_trace_flag = 0                            # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plotter_flag = 0                               # 0 - disable, 1 - plot, 2 - save, 3 - both
    self.plotter_switch = [6]                           # [] - none, [1], [1,2], [1,3], [99] - all
    self.plotter_mlp_flag = 0                           # 0 - disable, 1 - plot, 2 - save, 3 - both

    self.sensor_center_enable = True

    self.action_zero_is_allowed = False

    self.action_take_when = {'Always': 'True', 'i <= 24': 'i <= 24', 'i % 3 == 2': 'i % 3 == 2'}
    self.action_take = 'i % 3 == 2'

    self.linear_regression_calculation = 'old'

    self.plotter = plotter

    self.storage = storage

    self.road = road
    self.x = 0
    self.y = self.road.wall_center[0]
    self.sight = 400                                    # ennyit lát előre 300, 54, 154
    self.sight_center = 400                             # ennyit lát előre 150

    self.y_history  = []
    self.x_history  = []
    # self.y_center   = []
    self.y_center   = self.road.wall_center
    self.y_distance = []
    self.y_distance_real = []
    self.y_distance_predicted = []
    self.y_distance_predicted_inv = []
# Bevezetésre került a LinearRegression intercept nélkül
    self.regression = LinearRegression(fit_intercept=False)
# Bevezetésre került az MLPRegreression
    self.mlp = MLPRegressor(hidden_layer_sizes=(10, 5), # (10, 5)
                            activation='tanh', # relu, tanh, logistic
                            solver='adam',
                            batch_size='auto',
                            learning_rate_init=0.01,
                            max_iter=1,                           # incremental learning - one step
                            shuffle=False,                        # erre is oda kell figyelni
                            random_state=1,
                            verbose=True, warm_start=True,        # New test warm_start and verbose 9:19 alatt 719-ig, 600 5:34
                            momentum=0.9,
                            nesterovs_momentum=True,
                            early_stopping=True,
                            n_iter_no_change=98765000)

    self.x_minmaxscaler = MinMaxScaler(feature_range=(-1,1))       # A range amibe skálázunk (-1, 1)
    self.y_minmaxscaler = MinMaxScaler(feature_range=(-1,1))       # A range amibe skálázunk (-1, 1)

# v.New Regression
# Első körben viszzsaállítom az intercpetet
    self.regression_left = LinearRegression(fit_intercept=True)
#    self.regression_left = LinearRegression(fit_intercept=False)   # Kiiktattam az intercept-et
    self.regression_center = LinearRegression(fit_intercept=True)
#    self.regression_center = LinearRegression(fit_intercept=False) # Kiiktattam az intercept-et
    self.regression_right = LinearRegression(fit_intercept=True)
#    self.regression_right = LinearRegression(fit_intercept=False)  # Kiiktattam az intercept-et

    print('---------------------------- HELLO --------------------------')
    print('---------------------- ÚJ AUTO VAGYOK :)) -------------------')

    # data holders
    self.sensor_center = []
    self.sensor_left   = []
    self.sensor_right  = []
    self.before  = []
    self.after   = []

    # new v.25
    # model data holders
    self.regression_left_coef_history = []
    self.regression_center_coef_history = []
    self.regression_right_coef_history = []

    # error holder
    self.loss_holder = []

    # mlp_fit_evaluation_time_holder
    self.mlp_fit_evaluation_time_holder = []
    
    self.mesterseges_coutner = 0

    # logger helyett
    self.printer = printer


  def calculate_distances(self):
    
    #    k = self.x; d = 0
    #    while(k < self.x + self.sight_center):
    #      k += 1; d += 1
    #      self.distance_center_from_wall = d
    #      if(int(self.road.wall_left[k]) < self.y):
    #        self.printer.sr('Sensor center = ', self.distance_center_from_wall)
    #        break
    #      if(int(self.road.wall_right[k]) > self.y):
    #        self.printer.sr('Sensor center = ', self.distance_center_from_wall)
    #        break

    k = self.x; d = 0
    while(k < self.x + self.sight_center):
      k += 1; d += 1
      self.distance_center_from_wall = d
      # v.24 - new
      if( self.sensor_center_enable == True ):
        if(int(self.road.wall_left[k]) < self.y):
          self.printer.sr('Sensor center = ', self.distance_center_from_wall)
          break
        if(int(self.road.wall_right[k]) > self.y):
          self.printer.sr('Sensor center = ', self.distance_center_from_wall)
          break
      if( self.sensor_center_enable == False ):
        self.distance_center_from_wall = 0
        
    # Kiiktatom a középső szenzort -> egyszerűen beállítom az értékét 0-ra
    # self.distance_center_from_wall = 0


    #    k = self.x; d = 0
    #    while(k < self.x + self.sight):
    #      k += 1;  d += 1
    #      self.distance_left_from_wall = d
    #      if(int(self.road.wall_left[k]) < self.y + d):
    #        self.printer.sr('Sensor from left wall = ', self.distance_left_from_wall)
    #        break

    # Ehelyett most az van hogy nézzen simán oldalra
    self.distance_left_from_wall = self.y - self.road.wall_left[self.x]


    #    k = self.x; d = 0
    #    while(k < self.x + self.sight):
    #      k += 1; d += 1
    #      self.distance_right_from_wall = d
    #      if(int(self.road.wall_right[k]) > self.y - d):
    #        self.printer.sr('Sensor from right wall = ', self.distance_right_from_wall)
    #        break

    # Ehelyett most az van hogy nézzen simán oldalra
    self.distance_right_from_wall = self.road.wall_right[self.x] - self.y


    # ki kell kalkulálni a tényleges távolságot a ball és a jobb faltól
    # mert ezekre fogom tanítani a neurális hálót, ahol ezeket becsüljük
    # és a bemeneti változó a 3 szenzorból érkező adat lesz.
    # valójában azt mérjük, hogy milyen távolságra van az út közepétől

    # Úgy vettem észre, hogy sehol nem használom
    # ToDo: Sehol nem használom ki kell venni a kódból
    # self.distance_from_top     = abs(self.road.wall_left[self.x] - self.y)
    # self.distance_from_bottom  = abs(self.road.wall_right[self.x] - self.y)
    # self.printer.sr('most távolsagra van a felső faltól = ', self.distance_from_top)
    # self.printer.sr('most távolsagra van az alsó faltól = ', self.distance_from_bottom)


    self.printer.info('c-------------------------------------------------------------------')
    self.printer.info('self.x                             = ', self.x)

    # ezt az értéket fogom becsülni, a középértéktől való eltérés mértéke, ha pozitív akkor fölfelé, ha negatív akkor lefelé tér el
    self.vertical_distance_from_middle = self.y - self.road.wall_center[self.x]

    self.printer.info('self.vertical_distance_from_middle = ', self.vertical_distance_from_middle)

    self.printer.info('ezt fogjuk becsülni, ez a középértéktől való eltérés mértéke = ', self.vertical_distance_from_middle)
    self.printer.info('k-------------------------------------------------------------------')

    # de elötte szeretnék még valamit leellenőrizni
    # ezeknek a hossza nem fog megeggyezni a tényleges futások számával, hanem több lesz
    # (milyen jó lett volna erre egy teszt esetet írni és akkor test driven development lenne)

    self.printer.debug('\t\t\t ---------------- Teszt ----------------')
    self.printer.debug('\t\t\t len(self.y_distance)    = ', len(self.y_distance))
    self.printer.debug('\t\t\t len(self.sensor_left)   = ', len(self.sensor_left))
    self.printer.debug('\t\t\t len(self.sensor_center) = ', len(self.sensor_center))
    self.printer.debug('\t\t\t len(self.sensor_right)  = ', len(self.sensor_right))
    self.printer.debug('\t\t\t self.x                  = ', self.x)
    self.printer.debug('\t\t\t -------------- Teszt End --------------')


  def append(self):
    self.y_distance.append(self.vertical_distance_from_middle)

    self.sensor_left.append(self.distance_left_from_wall)
    self.sensor_center.append(self.distance_center_from_wall)
    self.sensor_right.append(self.distance_right_from_wall)

    self.printer.debug('\t\t\t ---------------- Append ----------------')
    self.printer.debug('\t\t\t len(self.y_distance)    = ', len(self.y_distance))
    self.printer.debug('\t\t\t len(self.sensor_left)   = ', len(self.sensor_left))
    self.printer.debug('\t\t\t len(self.sensor_center) = ', len(self.sensor_center))
    self.printer.debug('\t\t\t len(self.sensor_right)  = ', len(self.sensor_right))
    self.printer.debug('\t\t\t self.x                  = ', self.x)
    self.printer.debug('\t\t\t -------------- Append End --------------')



  def plot_history(self, flag, autoscale = True):
    if( flag != 0 ):
      plot_history(auto = self, flag = flag, autoscale = autoscale)

      # print(' --------------- plot --------------- ')


  def plot_history_fixed(self, flag, ymin, ymax, width, height):
    if( flag != 0 ):
      plot_history_fixed(self, flag, ymin, ymax, width, height)
    
      # print(' --------------- plot --------------- ')


  def plot_history_range(self, flag, start = 0, end = 9999999, autoscale = True):
    if( flag != 0 ):
      plot_history_range(auto = self, flag = flag, start = start, end = end, autoscale = autoscale)

      # print(' --------------- plot --------------- ')

# Bevezetésre került, elementi a képet
  def save_plots(self):
# y_distance vs y_distance_predicted
    plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted)
    plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real');
    plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
    plt.savefig('y_distance_vs_y_distance_predicted_{0:04}'.format(self.x)+'.png')
    plt.close()


# y_distance vs y_distance_predicted összes adaton
    X_test_full = np.array([self.sensor_left, self.sensor_center, self.sensor_right]).T
    _X_test_full = X_test_full
#    predicted_test_full = self.regression.predict(_X_test_full)
  # Lineáris regresszió helyett Neurális hálót használok
    _X_test_full_scaled = self.x_minmaxscaler.transform(_X_test_full)
    predicted_test_full = self.mlp.predict(_X_test_full_scaled)
  # ToDo : itt még lehet, hogy kéne transzformálni y-t is és az egészet visszatranszformálni eredeti értékére + ellenőrizni, hogy tulajdonképpen amikor skálázom az y-t akkor mi alapján skálázok
    predicted_test_full = self.y_minmaxscaler.inverse_transform(predicted_test_full.reshape(-1, 1))
    _y_test_full = np.array([self.y_distance]).T
    print(_y_test_full.shape)
    print(predicted_test_full.shape)
    plt.figure(figsize=(12, 5)); plt.scatter(_y_test_full, predicted_test_full, c='r');
    plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real');
    plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
    plt.savefig('y_distance_vs_y_distance_predicted_all_{0:04}'.format(self.x)+'.png')
    plt.close()


# y_distance vs y_distance_predicted összes adaton színezve
    _array_target = np.array([_y_test_full.ravel(), predicted_test_full.ravel(), np.arange(0, _y_test_full.shape[0], 1)]).T

    plt.figure(figsize=(12, 5)); plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2]);
    plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real');
    plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
    plt.savefig('y_distance_vs_y_distance_predicted_all_color_{0:04}'.format(self.x)+'.png')
    plt.close()


# y_distance vs y_distance_predicted összes adaton színezve vezető vonallal
    plt.figure(figsize=(12, 5)); ax = plt.axes(); ax.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2])
    ax.plot([-10, 2, 4, 10], [-10, 2, 4, 10]); ax.set_ylabel('y_distance_predicted'); ax.set_xlabel('y_distance_real');
    ax.set_title('#i = ' + str(self.x), fontsize=18, fontweight='bold')
    plt.savefig('y_distance_vs_y_distance_predicted_all_color_line_{0:04}'.format(self.x)+'.png')
    plt.close()


# Milyen kapcsolat van a bemenő adatok és a célváltozó között
    plt.figure(figsize=(12, 5)); plt.scatter(self.sensor_left, self.y_distance, c=_array_target[:,2]);
    plt.ylabel('self.y_distance'); plt.xlabel('self.sensor_left');
    plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
    plt.savefig('sensor_left_vs_y_distance_{0:04}'.format(self.x)+'.png')
    plt.close()


# y_distance vs y_distance_predicted összes adaton színezve vezető vonallal
    plt.figure(figsize=(12, 5)); ax = plt.axes(); ax.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2])
    ax.plot([-10, 2, 4, 10], [-10, 2, 4, 10]); ax.set_ylabel('y_distance_predicted'); ax.set_xlabel('y_distance_real');
    ax.set_ylim((-30, 30)); ax.set_xlim((-50, 50));
    ax.set_title('#i = ' + str(self.x), fontsize=18, fontweight='bold')
    plt.savefig('y_distance_vs_y_distance_predicted_all_color_line_fix_{0:04}'.format(self.x)+'.png')
    plt.close()

    print(' --- plots have been saved --- ')


  def plot_investigation(self, _y_test_full, predicted_test_full, flag):

    if( flag != 0 ):

#      plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted);
#      plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
#      plt.savefig('yDistance_vs_yDistance_predicted_typeAAAA_{0:04}'.format(self.x)+'.png');
#      plt.show();
#      plt.close();

#      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted);
#      plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
#      # white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.y_distance_real))); plt.legend(handles=[white_patch])
#      if( flag == 1 or flag == 3 ): plt.show();
#      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_typeCCCC_{0:04}'.format(self.x)+'.png'); plt.close(fig);

      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között (csak a tanítás után)
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted);
      plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.y_distance_real))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type0_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # mmmmmmmmmmmmmmmm
      # ez itt kulcsfontosságú lesz.
      # az alap problémám az volt vele, hogy a függőleges tengelyen lévő adatok nem mormalizáltak
      # a vizsszíintes tengelyen viszont a neurális háló utáni becsült értékek normalizált formában jelennek meg
      # ezért amelett, hogy első körben meghagyom a fenti plotot kell csinálnom egy olyat amin a neurális háló által becsült értékek
      # vissza vannak transzformálva

      # 1)
      #
      # Ebben az a csalóka, hogy becsült értékeket eltároljuk
      # itt viszont egy olyan visszatranszformációt hajtok végre rajtuk
      # amiközben már változott a <<self.y_minmaxscaler>>
      # tehát a visszatranszformáció igazából nem lesz helyes
      #
      # ahol y vissza van transformálva eredeti formájára
      print('type(self.y_distance_predicted) = ', type(self.y_distance_predicted))
      inv_y_distance_predicted = self.y_minmaxscaler.inverse_transform(self.y_distance_predicted)
      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között (csak a tanítás után)
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, inv_y_distance_predicted);
      plt.ylabel('y_distance_predicted_inv (wrong)'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.y_distance_real))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_typeWrong_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # 2)
      #
      # Elvileg ez a helyes - de a fentit meghagyom hogy lássam a különbséget
      # ahol a <<y_distance_predicted>> változó előáll ott csinálok rajta gyorsan egy visszatranszformációt
      # és azt is eltárolom egy listában
      # majd pedig azt jelenítem itt meg (sokkal tisztább, nehogy már egy plot fügvényben legyen adat transzformáció)
      #
      # így ugyanis akkor áll elő a visszatranszformáció amikor még ugyan azokat az adatokat kapta meg a <<self.y_minmaxscaler>>
      #
      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      # (csak a tanítás után)
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted_inv);
      plt.ylabel('y_distance_predicted_inv (correct)'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.y_distance_real))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_typeCorrect_bw_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # X) ugyan ez csak az idő színnel kiegészítve
      _time = np.array([np.arange(0, len(self.y_distance_predicted_inv), 1)]).T
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.y_distance_real, self.y_distance_predicted_inv, c=_time);
      plt.ylabel('y_distance_predicted_inv (correct)'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.y_distance_real))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_typeCorrect_col_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');


      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      fig = plt.figure(figsize=(12, 5)); plt.scatter(_y_test_full, predicted_test_full, c='r');
      plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      # plt.ylabel(r'$\int\ Y^2\ dt\ \ [V^2 s]$')
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_y_test_full))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type1_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      _array_target = np.array([_y_test_full.ravel(), predicted_test_full.ravel(), np.arange(0, _y_test_full.shape[0], 1)]).T

      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      fig = plt.figure(figsize=(12, 5)); plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2]);
      plt.ylabel('y_distance_predicted'); plt.xlabel('y_distance_real'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type2_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      fig = plt.figure(figsize=(12, 5)); ax = plt.axes(); ax.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2])
      ax.plot([-10, 2, 4, 10], [-10, 2, 4, 10]); ax.set_ylabel('y_distance_predicted'); ax.set_xlabel('y_distance_real');
      ax.set_title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type3_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      fig = plt.figure(figsize=(12, 5)); ax = plt.axes(); ax.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2])
      ax.plot([-10, 2, 4, 10], [-10, 2, 4, 10]); ax.set_ylabel('y_distance_predicted'); ax.set_xlabel('y_distance_real');
      ax.set_ylim((-30, 30)); ax.set_xlim((-50, 50)); ax.set_title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type4_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Milyen kapcsolat van a középponttól vett távolság és ugyan ennek a változónak a neurális hálóval becsült értéke között
      fig = plt.figure(figsize=(12, 5)); ax = plt.axes(); ax.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2])
      ax.plot([-20, 2, 4, 20], [-20, 2, 4, 20]); ax.set_ylabel('y_distance_predicted'); ax.set_xlabel('y_distance_real');
      ax.set_ylim((-60, 60)); ax.set_xlim((-60, 60)); ax.set_title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('yDistance_vs_yDistance_predicted_type5_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');


  def plot_investigation_senors(self, _y_test_full, predicted_test_full, flag):

    if( flag != 0 ):

      _array_target = np.array([_y_test_full.ravel(), predicted_test_full.ravel(), np.arange(0, _y_test_full.shape[0], 1)]).T

      # Milyen kapcsolat van a bal oldali szenzor <<bemenő adat>> és a célváltozó között
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.sensor_left, self.y_distance, c=_array_target[:,2]);
      plt.ylabel('self.y_distance'); plt.xlabel('self.sensor_left'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.sensor_left))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('sensorLeft_vs_yDistance_v1_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      self.printer.info('len(self.sensor_left) = ', len(self.sensor_left))
      self.printer.info('len(self.y_distance) = ', len(self.y_distance))

      # Milyen kapcsolat van a közéső szenzor <<bemenő adat>> és a célváltozó között
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.sensor_center, self.y_distance, c=_array_target[:,2]);
      plt.ylabel('self.y_distance'); plt.xlabel('self.sensor_center'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.sensor_center))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('sensorCenter_vs_yDistance_v1_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Milyen kapcsolat van a jobb oldali szenzor <<bemenő adat>> és a célváltozó között
      fig = plt.figure(figsize=(12, 5)); plt.scatter(self.sensor_right, self.y_distance, c=_array_target[:,2]);
      plt.ylabel('self.y_distance'); plt.xlabel('self.sensor_right'); plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(self.sensor_right))); plt.legend(handles=[white_patch])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig('sensorRight_vs_yDistance_v1_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');



  def plot_before_after_sensor_estimation_in_one_chart(self, _y_sensor, _predicted_sensor, y_delta, name, flag):

    if( flag != 0 ):

      fileName = 'sensor' + name.capitalize() + 'AfterScaled_vs_sensor' + name.capitalize() + 'PredictedAfterScaled_S1'

      fig = plt.figure(figsize=(18, 7.5));
      # plt.figure(figsize=(18, 7.5));
      
      # color -> y_delta
      plt.subplot(1, 3, 1)
      plt.scatter(_y_sensor, _predicted_sensor, c=y_delta); plt.ylabel('_predicted_' + name); plt.xlabel('_true_' + name);
      plt.title('#i = ' + str(self.x));
      cmap = mpl.cm.viridis
      # bounds = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
      bounds = np.arange(y_delta.min(), y_delta.max(), 1)
      if( bounds.size < 3 ): bounds = [-1, 0, 1]
      # print('bounds = ', bounds)
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='horizontal',
             label='Elmozdítás mértéke');

      # nem biztos, hogy kell mellé az idő is, de elvileg az még hiányzik
      _array_target = np.array([_y_sensor.ravel(), _predicted_sensor.ravel(), y_delta.ravel(), np.arange(0, y_delta.shape[0], 1)]).T

      # [[_array_target]] [_y_sensor, _predicted_sensor, y_delta, time] pl [_y_left, _predicted_left, y_delta, time]
      # color -> time
      plt.subplot(1, 3, 2)
      plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,3]); plt.ylabel('_predicted_' + name); plt.xlabel('_true_' + name);
      plt.title('#i = ' + str(self.x));
      plt.colorbar(orientation='horizontal', label='Time');

      # [[_array_target]] [_y_sensor, _predicted_sensor, y_delta, time] pl [_y_left, _predicted_left, y_delta, time]
      # color -> time, size = y_delta
      plt.subplot(1, 3, 3)
      plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,3], s=_array_target[:,2]+10); plt.ylabel('_predicted_' + name); plt.xlabel('_true_' + name);
      plt.title('#i = ' + str(self.x));
      plt.colorbar(orientation='horizontal', label='Time');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');


  def plot_before_after_sensor_values(self, _array_target, name, flag):

    if( flag != 0 ):

      fileName = 'sensor' + name.capitalize() + 'BeforeScaled_vs_sensor' + name.capitalize() + 'AfterScaled'

      # Mi a kapcsolat a before after sesoros adatok között [[ez nem a becslés, hanem a nyers adatok]]
      # _array_target = [[before_array[:,1](sensor), after_array[:,1](sensor), y_delta{action}, time]]
      print(' ---------- plot scatter plot for before after value with time {color} 2 ----------------')

      fig = plt.figure(figsize=(6.25, 5));
      plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,3]);
      plt.ylabel('after'); plt.xlabel('before');
      plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      __x_max = _array_target[:,0].max();
      __x_min = _array_target[:,0].min();
      __y_max = _array_target[:,1].max();
      __y_min = _array_target[:,1].min();
      __x_cen = __x_max + ((__x_max - __x_min) * 0.1);
      __y_cen = (__y_max + __y_min)/2;
      plt.text(__x_cen, __y_cen, name, rotation='vertical', horizontalalignment='center', verticalalignment='center');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      plt.colorbar(orientation='vertical', label='time');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_v1_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Mi a kapcsolat a before after sesoros adatok között [[ez nem a becslés, hanem a nyers adatok]]
      print(' ---------- plot scatter plot for before after value with time {color} and action {size} 2 ----------------')
      fig = plt.figure(figsize=(6.25, 5));
      size = _array_target[:,2]
      size = np.abs(size) * 4 + 3
      plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,3], s=size);
      plt.ylabel('after'); plt.xlabel('before');
      plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      plt.text(__x_cen, __y_cen, name, rotation='vertical', horizontalalignment='center', verticalalignment='center');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      plt.colorbar(orientation='vertical', label='time');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_v2_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

      # Mi a kapcsolat a before after sesoros adatok között [[ez nem a becslés, hanem a nyers adatok]]
      print(' ---------- plot scatter plot for before after value with time and action {color} 2 ----------------')
      fig = plt.figure(figsize=(6.25, 5));
      plt.scatter(_array_target[:,0], _array_target[:,1], c=_array_target[:,2]);
      plt.ylabel('after'); plt.xlabel('before');
      plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      plt.text(__x_cen, __y_cen, name, rotation='vertical', horizontalalignment='center', verticalalignment='center');
      white_patch = mpatches.Patch(color='white', label='number of observation = ' + str(len(_array_target[:,0]))); plt.legend(handles=[white_patch])
      cmap = mpl.cm.viridis
      bounds = np.arange(_array_target[:,2].min(), _array_target[:,2].max() + 1, 1)
      if( bounds.size < 3 ): bounds = [-1, 0, 1]
      # bounds = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
      # print('bounds = ', bounds)
      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
      plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='vertical',
             label='Elmozdítás mértéke');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_v3_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');

  # Itt volt egy függvény a plot_state_space_discover(self, flag) amit szétbontottam négy felé.
  # https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
  # @deprecated
  def plot_state_space_discover(self, flag):
    '''Csak azért van hogy régi notebookok ne törjenek el'''
    self.plot_state_space_discover_type1(flag)
    self.plot_state_space_discover_type2(flag)
    self.plot_state_space_discover_type3(flag)
    self.plot_state_space_discover_type4(flag)

  def plot_state_space_discover_type1(self, flag):

    # Az adatok nem a before afterből kellenek nekünk, hanem
    # self.y_distance;    # self.sensor_left;    # self.sensor_center;    # self.sensor_right;    # time -> create

    if( flag != 0 ):

      fileName = 'state_space_discover'

      szin = np.arange(len(self.sensor_right))

      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(projection='3d')
      ax.scatter(self.sensor_left, self.sensor_right, self.sensor_center, c=szin)
      ax.set_xlabel('sensor left')
      ax.set_ylabel('sensor right')
      ax.set_zlabel('sensor center')
      ax.invert_xaxis()
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightCenter_3D_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');


  def plot_state_space_discover_type2(self, flag):

    # Az adatok nem a before afterből kellenek nekünk, hanem
    # self.y_distance;    # self.sensor_left;    # self.sensor_center;    # self.sensor_right;    # time -> create

    if( flag != 0 ):

      fileName = 'state_space_discover'

      szin = np.arange(len(self.sensor_right))

      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(projection='3d')
      ax.scatter(self.sensor_left, self.sensor_right, self.y_distance, c=szin)
      ax.set_xlabel('sensor left')
      ax.set_ylabel('sensor right')
      ax.set_zlabel('y_distance')
      # ax.invert_xaxis()
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');


  def plot_state_space_discover_type3(self, flag):

    # Az adatok nem a before afterből kellenek nekünk, hanem
    # self.y_distance;    # self.sensor_left;    # self.sensor_center;    # self.sensor_right;    # time -> create

    if( flag != 0 ):

      fileName = 'state_space_discover'

      szin = np.arange(len(self.sensor_right))

      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(projection='3d')
      plot = ax.scatter(self.sensor_left, self.sensor_right, self.y_distance, c=szin, cmap='winter')
      ax.set_xlabel('sensor left')
      ax.set_ylabel('sensor right')
      ax.set_zlabel('y_distance')
      ax.invert_xaxis()
      # Get rid of colored axes planes
      # First remove fill
      ax.xaxis.pane.fill = False
      ax.yaxis.pane.fill = False
      ax.zaxis.pane.fill = False
      # Now set color to white (or whatever is "invisible")
      ax.xaxis.pane.set_edgecolor('w')
      ax.yaxis.pane.set_edgecolor('w')
      ax.zaxis.pane.set_edgecolor('w')
      # Bonus: To get rid of the grid as well:
      ax.grid(False)
      # Colorbar:
      # Add colorbar
      cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
      # cbar.set_ticks([0, 50, 100, 150, 200])
      # cbar.set_ticklabels(['0', '50', '100', '150', '200 nm'])
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_WhitoutBorder_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');


  def plot_state_space_discover_type4(self, flag):

    # Az adatok nem a before afterből kellenek nekünk, hanem
    # self.y_distance;    # self.sensor_left;    # self.sensor_center;    # self.sensor_right;    # time -> create

    if( flag != 0 ):

      fileName = 'state_space_discover'

      szin = np.arange(len(self.sensor_right))

      fig = plt.figure(figsize=(7.5, 6))
      plt.scatter(self.sensor_left, self.sensor_right, c=self.y_distance)
      plt.ylabel('sensor_right'); plt.xlabel('sensor_left');
      plt.colorbar(orientation='vertical', label='y_distance');
      plt.title('#i = ' + str(self.x))
      # plt.title('#i = ' + str(self.x), fontsize=18, fontweight='bold');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_2D_{0:04}'.format(self.x)+'.png'); plt.close(fig); plt.close('all'); fig.clf(); plt.close('all');



  def plot_trace(self, freq, flag):
    if( flag != 0 ):
      plot_trace(auto = self, freq = freq, flag = flag)

      # print(' --------------- plot --------------- ')






# ----------------------------------------------------------------------------------------------------------------------------


  def run(self, run_length, silent = False):
    for i in range(0, run_length, 1):
      #self.printer.util(BColors.WARNING + "Warning: No active frommets remain. Continue?" + BColors.ENDC)
      self.printer.util(BColors.WARNING + "\n# A run ciklus eleje ----------------------------------------------------------------------------------------------------------" + BColors.ENDC)
      # self.printer.util('\n# A run ciklus eleje ----------------------------------------------------------------------------------------------------------')
      self.printer.util('# i = ', i)
      _summary_mlp_prediction_was_taken = 0
      _summary_mlp_fit_was_taken = 0
      _summary_mesterseges_mozgatas = 0
      _summary_action_was_taken = 0

# Beállítja az x értékét az éppen aktuális ciklusváltozó értékére
      self.x = i
# Kiszámoja a szenzoroknak a faltól mért távolságát
      self.calculate_distances()
# Eltárolja a kiszámolt értékeket
      self.append()

# Csak néha plottoljunk ne mindíg, egyébként a függvény is megkapja hogy mikor plottoljon
      if ( i % self.plot_frequency == 0 ):
        
        # Show history plot - Save history plot
        self.plot_history(self.plot_history_flag)
        
        # New
        self.plot_trace(self.plot_frequency, self.plot_trace_flag)





# Itt kezdődik a lényeg
      if ( i >= 0 ):

        # --------------------------------------- A NEURÁLIS HÁLÓ TANÍTÁSA (1) ---------------------------------------
        
        if ( i % 3 == 0 and i >= 12 ):

          self.printer.info('----------------------- A NEURÁLIS HÁLÓ TANÍTÁSA --------------------------')
          self.printer.info('------------------------------ IF i % 3 == 0 ------------------------------')
          self.printer.info('# i = ', i)
          self.printer.util('# i = ', i)
          self.printer.info('# 1. számú tanulás. Mi a kapcsolat a szenzoros adatok és aközött, hogy az út melyik részén van az autó (micadoban ez az NN)')
          X = np.array([self.sensor_left, self.sensor_center, self.sensor_right]).T
          y = np.array([self.y_distance]).T
          self.printer.debug('X.shape = ', X.shape)
          self.printer.debug('y.shape = ', y.shape)
          self.printer.debug('X       = ', X)
          self.printer.debug('y       = ', y)
          
          _summary_mlp_fit_was_taken = 1

          self.x_minmaxscaler.fit(X)
          self.y_minmaxscaler.fit(y)
          X_scaled = self.x_minmaxscaler.transform(X)
          y_scaled = self.y_minmaxscaler.transform(y)
          self.printer.debug('---------------------')
          self.printer.debug('X.max = ', X.max())
          self.printer.debug('X.min = ', X.min())
          self.printer.debug('y.max = ', y.max())
          self.printer.debug('y.min = ', y.min())
          self.printer.debug('---------------------')
          self.printer.debug('X_scaled.shape = ', X_scaled.shape)
          self.printer.debug('y_scaled.shape = ', y_scaled.shape)
          self.printer.info('---------------------')
          self.printer.info('X_scaled.max = ', X_scaled.max())
          self.printer.info('X_scaled.min = ', X_scaled.min())
          self.printer.info('y_scaled.max = ', y_scaled.max())
          self.printer.info('y_scaled.min = ', y_scaled.min())
          self.printer.info('---------------------')
          mlp_fit_time_start = time.time()
          self.mlp.fit(X_scaled, y_scaled)
          mlp_fit_time = time.time() - mlp_fit_time_start
          self.mlp_fit_evaluation_time_holder.append(mlp_fit_time)

# Ha olyanunk van plottoljunk
          self.plotter.plot_mlp(mlp = self.mlp ,flag = self.plotter_mlp_flag)


        # --------------------------------------- A NEURÁLIS HÁLÓ MINŐSÉGÉNEK VISSZAMÉRÉSE, TESZTELÉSE (2) ---------------------------------------

        if( i % 3 == 1 and i >= 22 ):

          self.printer.info('---------------- A NEURÁLIS HÁLÓ MINŐSÉGÉNEK VISSZAMÉRÉSE -----------------')
          self.printer.info('------------------------------ IF i % 3 == 1 ------------------------------')
          self.printer.info('# i = ', i)
          self.printer.info('# 2. az 1. pontban megtanult modell alapján teszünk egy becslést - tulajdonképpen ezzel mérem az 1. modell jóságát, ez a lépés ezt szolgálja')
          X_test = np.array([self.sensor_left, self.sensor_center, self.sensor_right]).T
          _X_test = np.array([X_test[-1,:].reshape(-1,1)])
          _X_test = np.array([X_test[-1,:]])
          self.printer.info('actual _X_test = ', _X_test)

          _X_test_scaled = self.x_minmaxscaler.transform(_X_test)
          predicted_test = self.mlp.predict(_X_test_scaled)
# ToDo : Fontos lenne visszatranszformálni a predicted_test értéket mielőtt belekerül az archivumba
          _summary_mlp_prediction_was_taken = 1
          self.y_distance_real.append(self.y_distance[-1])
          self.y_distance_predicted.append(predicted_test)
# ToDo : Ellenőrizni, hogy közben nem változott-e a <<self.y_minmaxscaler>>
          predicted_test_inv = self.y_minmaxscaler.inverse_transform(predicted_test.reshape(-1, 1)).flatten()
          self.y_distance_predicted_inv.append(predicted_test_inv)

          self.printer.investigation('actual predicted_test = ', predicted_test)
          self.printer.investigation('actual self.y_distance[-1] = ', self.y_distance[-1])
          self.printer.investigation('actual predicted_test_inv = ', predicted_test_inv)
          self.printer.info('len(self.y_distance_real)      = ', len(self.y_distance_real))
          self.printer.investigation('len(self.y_distance_predicted) = ', len(self.y_distance_predicted))
          self.printer.investigation('len(self.y_distance_predicted_inv) = ', len(self.y_distance_predicted_inv))
          self.printer.debug('self.y_distance_real = \n', self.y_distance_real)
          self.printer.debug('self.y_distance_predicted = \n', self.y_distance_predicted)
          self.printer.debug('self.y_distance_perdicted_inv = \n', self.y_distance_predicted_inv)


# Plot : Minden 32-ik lépésbén kiplottoljuk a Neurális háló álltal előrejelzett és a tényleges adatok közötti kapcsolatot
# Ezzel a felétellel az a baj, hogy benne van egy másik if-ben ami azt mojda ki, hogyha i % 3 == 1
# Vagyi nem minden 32-ik lépésben plottolunk

# Lecseréltem ezt plot_frequency
          if( i % self.plot_detailed_frequency == 0 ):
            
            # korábban csak azokat az adatokat plottoltam amik a tanulás után lettek visszamérve, de nézzük meg a teljes adatsoron
            X_test_full = np.array([self.sensor_left, self.sensor_center, self.sensor_right]).T
            _X_test_full = X_test_full
            _X_test_full_scaled = self.x_minmaxscaler.transform(_X_test_full)
            predicted_test_full = self.mlp.predict(_X_test_full_scaled)
# ToDo : itt még lehet, hogy kéne transzformálni y-t is és az egészet visszatranszformálni eredeti értékére + ellenőrizni, hogy tulajdonképpen amikor skálázom az y-t akkor mi alapján skálázok
            predicted_test_full = self.y_minmaxscaler.inverse_transform(predicted_test_full.reshape(-1, 1))
            _y_test_full = np.array([self.y_distance]).T
            self.printer.info('_y_test_full.shape = ', _y_test_full.shape)
            self.printer.info('predicted_test_full.shape = ', predicted_test_full.shape)


# Plot
# (flag 0 = disable, flag 1 = plot, 2 = save, 3 = both)
            # Vizsgáljuk meg, hogy milyen kapcsolat van a becsült és a valós érték között
            self.plot_investigation(_y_test_full, predicted_test_full, self.plot_investigation_flag)

            # Illetve, hogy miyen kapcsolat van a szenzorok értékei és a becsült változó között (flag 0 = disable, flag 1 = plot, 2 = save, 3 = both)
            self.plot_investigation_senors(_y_test_full, predicted_test_full, self.plot_investigation_flag)



        # ----------------------------------------- MESTERSÉGES MOZGATÁS (3) -----------------------------------------

        # most jön az, hogy véletlenszerűen kell egyet ugrania fel, vagy le
        # ez felel meg a before after dologonak
        # az így létrejött adatokat is el kell tárolni úgy mint
        # mi volt a szezoros adat before
        # mi lett a szenzoros adat after
        # mi volt az y before, mi lett az y after
        # mivel mindíg egyet fogunk csak lépni, ezért a dif mindíg egy lesz
        # de ezt számítani kell, mivel a későbbiek folyamán lehet, hogy többet is fog lépni

        # ToDo. Azt hiszem ezt a beállítást szeretném kivezetni.

        # TesztTesztTeszt
        # if( i % 3 == 2 ):
        # if( i % 3 == 2 and i <=24 ):
        # if( i % 3 == -1 ): # tehát soha

        # hhhhh

        if eval(self.action_take):
            
          self.printer.info('-------------------------- MESTERSÉGES MOZGATÁS ---------------------------')
          self.printer.info('------------------------------ IF i % 3 == 2 ------------------------------')
          self.printer.info('# i = ', i)
          self.printer.info('# 3. véletlenszerűen változtatok az autó pozicióján -> ebből állnak elő a before after adatok')

          self.printer.info('self.y before move = ', self.y)

          _summary_mesterseges_mozgatas = 1
          
          if( self.mesterseges_coutner == 0 ):                             # Első lépésben fel
            self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.y = self.y + 1
            self.printer.info('artificial move -> up first')
            self.calculate_distances()
            self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.mesterseges_coutner = 1

          elif( self.mesterseges_coutner == 1 ):                           # Második lépésben le
            self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.y = self.y - 1
            self.printer.info('artificial move -> down first')
            self.calculate_distances()
            self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.mesterseges_coutner = 2

          elif( self.mesterseges_coutner == 2 ):                           # Harmadik lépésben le
            self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.y = self.y - 1
            self.printer.info('artificial move -> down second')
            self.calculate_distances()
            self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.mesterseges_coutner = 3

          elif( self.mesterseges_coutner == 3 ):                           # Negyedik lépésben fel
            self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.y = self.y + 1
            self.printer.info('artificial move -> up second')
            self.calculate_distances()
            self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
            self.mesterseges_coutner = 0

          else:
            self.printer.info('semmi\n\n\n\n')

          self.printer.info('self.y after move = ', self.y)


        # ------------------------------------------------ ACTION (X) ------------------------------------------------

        # itt van egy érdekesség amit csak magamnak írok fel ezen a ponton lépünk ki a három if ágból - if( i % 3 == _ )
        # ez azért fontos, mert ami itt következik az mindíg lefut
        # akár volt neurális háló tanítás
        # akár volt neurális háló predikció visszamérése
        # akár volt mesterséges mozgatás

        # Felmerül a kérdés de csak felmerül, hogy nem lehet-e az, hogy ezt csak akkor kéne elvégezni amikor nincs neurális
        # háló tanítás és nincs mesterséges mozgatás sem. Ezt csak egy kísérlet ereéig ki kéne próbálni (if i % 3 == 1) ->
        # vagyis amikor az nn predikció mérése történik


        # version 20. -> if( i % 3 == 0 ); version 22. -> if( i % 3 == 1 )

        # atcion változó fogja tárolni, hogy mi lenne az optimizer szerint a helyes döntés -> fontos, hogy ezt a döntést meg is lépi
        action = 0
        if( i % 3 == 0 ):

          # ez az ág csak akkor fut le, ha már van elég before-after adatunk,
          # amíg nincs, addig nem csinál semmilyen kiértékelést, nem hoz döntést

          if( len(self.before) > 9 ):

            self.printer.info('------------------------------------- ACTION ----------------------------------------')
            self.printer.info('------------------------------ IF len(self.before) > 9 ------------------------------')
            self.printer.info('\n')
            self.printer.info('  Ha már van elég before after adatunk')
            self.printer.info('# 3. Tanulás itt kerül kiszámításra a lineáris regresszió minden egyes metrikára')
            # minden egyes szezor adatára el kell készítenünk azt a lineráis regressziós modelt ami megmondja, hogy mi lenne a szenzor értéke, ha 1, 2, 3, ... n lépéssel elvinnénk a kocsit

            # oké megvan a before és megvan az after (self.y, left, center, right)
            # a before és az after array egyébként úgy épül fel, hogy a sorok a megfigyelések
            # 0-ik oszlop !!! Nem az ót közepétől vett eltérés mértéke, hanem az Y tengelyen mért távolság !!!
            # 1    oszlop sensor_left
            # 2    oszlop sensor_center
            # 3    oszlop sensor_right
            before_array = np.array(self.before)
            after_array  = np.array(self.after)
            y_delta = after_array[:,0] - before_array[:,0]
            delta_array = after_array - before_array
            # későbbi elemzés céljából elteszem
            self.delta_array = delta_array
            # hhh
            self.printer.ba('')
            self.printer.ba('------------- hogy a picsába van az, hogy az elmozdulás mértékeként hivatkozok rá és az értéke néha 0 -------')
            self.printer.ba('-------------                     az y_delta változóról van szó                                --------------')
            self.printer.ba('------------- azóta a self.action_zero_is_allowed = False kapcsolóval ki lehet zárni ezt a mechanizmust -----')
            self.printer.ba('------------- de ellenőzrés képpen itt hagyom ezt a megjegyzést emlékeztetőnek ------------------------------')
            
            self.printer.ba('\n----------------------- Before After Dataset Monitoring Block -----------------------')
            self.printer.ba('\ny_delta = \n')
            self.printer.ba(y_delta)
            self.printer.ba('before_array.shape = ', before_array.shape)
            self.printer.ba('after_array.shape  = ', after_array.shape)
            self.printer.ba('before_array = \n')
            self.printer.ba(before_array)
            self.printer.ba('after_array = \n')
            self.printer.ba(after_array)
            self.printer.ba('\ndelta_array = \n')
            self.printer.ba(delta_array)
            self.printer.ba('-----------------------------------------------------------------------------\n')
            


            # képlet szerint sensor_after' = w0 + w1 * sensor_before + w2 * delta_y
  # ToDo a helyes képlet nem ez --------> ki kell javítani
            # a sensor_after és a sensor_befor érték világs
            # a delta_y azt fejezi ki, hogy mi volt az autó Y tengelyen mért távolságában megvigyelhető elétrés
            # << a skálázási logikában ez a fel le skálázás mértéke >>
            #
            # magyarul azt akarjuk megbecsülni <<sesor_after>> hogy hogyan állítható elő ez az értéke a sensor_before <<jelenlegi,
            # vagy elmozdítás elötti értékéből>> és az Y tengelyen vett elmozdulás mértékéből <<delta_y>>
            # [[Gondolom világos de azért leírom, hogy nem az autó Y tengelyen vett poziciójábaól]]
            # [[Hanem abból, hogy mekkora volt az elmozdulása az Y tengelyen]]

            self.printer.lr('# Linear Regression Learning --------------------------------------------------------------------------------------')
            self.printer.lr('\t\t # Linear Regression Training Results -------------------------------------------------------------------------')

  # --------------
  # --------------
  # -------------- left
  # Ki akarom majd vezetni az egész számítást egy osztályba de addig is hhh
            if( self.linear_regression_calculation == 'old'):
              _X_left = np.array([before_array[:,1], delta_array[:,0]]).T # left és delta_y (before)

            if( self.linear_regression_calculation == 'new'):
              _egy_left     = np.array([before_array[:,1] * before_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              _ketto_left   = np.array([before_array[:,1] * delta_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              _X_left_proba = np.stack((_egy_left.flatten(), _ketto_left.flatten()), axis=1)
              _X_left = _X_left_proba
              # nem csak itt kell átírni hanem ahol a lehetséges actionökre is kiszámolja

            #> _y_left a becsült érték pedig a left sesor elmozdulás után mért értéke
            _y_left = after = after_array[:,1].reshape(-1, 1)
            self.regression_left.fit(_X_left, _y_left)
            # print('\t\t _X_left <<sensor before, y elmozdulás mértéke>>   = \n', _X_left)
            # print('\t\t _y_left <<sensor after az érték amit becsülnünk>> = \n', _y_left)
            self.printer.lr('\t\t ------------------------------- valyon mennyire jó a left   metrikának a becslése -----------------------------')
            self.printer.lr('\t\t self.regression_left.coef_ = ', self.regression_left.coef_)
            self.printer.lr('\t\t self.regression_left.intercept_ = ', self.regression_left.intercept_)
            #> _predicted_left lesz a bemenete a neurális hálónak
            #  nem a mostani formájában mert itt a tényleges le fel skálázási adatok alapján tanítottuk meg a lineáris regressziós modelt
            #  arra, hogy milyen összefüggés van a (1) skálázás elötti szenzoros adat értéke (2) az elmozdulás mértéke (3) és az így kapott
            #  új szenzoros érték között.
            #
            #  Miután előállt a modellünk <<regression_left>> ezzel fogjuk kiszámolni, hogy mi lenne a szenzor új értéke {+1, +2, +3, ..}
            #  elmozdítás esetén. -> Majd az így kapott értékeket pakoljuk be egyenként a neurális hálóba és számojuk ki, hogy mi lenne
            #  az így kapott Y tengelyen mért érték -> Majd pedig ennek alapján választjuk ki azt, amelyikkel a legközelebb tudunk
            #  jutni a kívánt célhoz

            _predicted_left = self.regression_left.predict(_X_left)

            #> Tehát a fenti <<_predicted_left>> változót csak azért hoztam létre, hogy vizsgálni tudjam, mennyire jól ragadta meg
            #  before<->after kapcsolatot leíró modell a kapcsolatot és mennyire jól képes becsülni az új értéket.
            #  [[tulajdonképpen ez itt egy dummy változó amit csak analízisre használok]]

            # mennyire jó a left szenzor before after becslése

            # Plot
            # (flag 1 = plot, 2 = save, 3 = both)
            # Vizsgáljuk meg, hogy milyen kapcsolat van a [...]
            # Ez egy nagyon érdekes Grafikon
            # Még nekem is barátkoznom kell az értelmezésével
            # Ezért erről később írok

            # Lecseréltem ezt plot_frequency
            if( i % (3 * self.plot_frequency) == 0 ):
              self.plot_before_after_sensor_estimation_in_one_chart(_y_left, _predicted_left, y_delta, 'left', self.plot_before_after_sensor_estimation_flag)
              # Erre
              # job_for_1A = multiprocessing.Process(target=self.plot_before_after_sensor_estimation_in_one_chart,args=(_y_left, _predicted_left, y_delta, 'left', self.plot_before_after_sensor_estimation_flag))
              # job_for_1A.start()

            self.printer.ba('_X_left << az a változó csomag ami adott szenzorra a sensor before értékét és az Y tengelyen vett elmozdulás mértéke >>\n', _X_left)
            self.printer.ba('_y_left << az a változó vector ami egy elmozdítás után mért szenzor értékét hordozza [ilyere változott] az elmozdítás után>>\n', _y_left)
            self.printer.ba('_predicted_left << az a változó vector amit az _X_left becsült _y_left értékeire [ez maga a becslést tartalmazó adatsor]>>\n', _predicted_left)
            
            # Arra vagyok kiváncsi, hogy melyik az _X_left-ben a változás mértéke
            self.printer.ba('_X_left.shape << ellenőrzés arra, hogy a két adacsomag hossaz megegyezik-e >>         = ', _X_left.shape)
            self.printer.ba('_predicted_left.shape << ellenőrzés arra, hogy a két adacsomag hossaz megegyezik-e >> = ', _predicted_left.shape)

            # Eddig egy konkrét sensor skálázás utáni értéke és skálázás utáni értéke becslés alapján közötti kapcsolatot vizsgáltunk
            # Most vizsgáljuk meg csak a maga egyszerűsgében azt, hogy milyen kapcsolat van a skálás elötti valós és a skálázás utáni valós érté között
            # kiplottolom a before after adatokat egy konkrét szenzor értékeire

            # [[ez a változó is csak azért kell, hogy lássam hogy áll az aktuális sensor before after érteke]]
            # [[sensor link]]
            # [[before_array[:,1](left), after_array[:,1](left), y_delta{action}, time]]
            _array_target_left = np.array([before_array[:,1].ravel(), after_array[:,1].ravel(), y_delta.ravel(), np.arange(0, after_array.shape[0], 1)]).T

            # Plot
            # (flag 0 = disable, 1 = plot, 2 = save, 3 = both)
            # Vizsgáljuk meg, hogy milyen kapcsolat van a [...]
            # bal szenzor skálázás elötti és a bal szenzor skálázás utáni értéke között
            # Lecseréltem ezt
            if( i % (3 * self.plot_frequency) == 0 ):
              self.plot_before_after_sensor_values(_array_target_left, 'left', self.plot_before_after_sensor_values_flag)
            # https://stackoverflow.com/questions/19662906/plotting-with-matplotlib-in-threads
            # Erre
            # job_for_1 = multiprocessing.Process(target=self.plot_before_after_sensor_values,args=(_array_target_left, 'left', self.plot_before_after_sensor_values_flag))
            # job_for_1.start()
            
            self.printer.ba('before_array.shape = ', before_array[:,1].shape)
            self.printer.ba('after_array.shape  = ', after_array[:,1].shape)
            self.printer.ba('array_target_left  = \n', _array_target_left)

  # --------------
  # --------------
  # -------------- center
  # Ki akarom majd vezetni az egész számítást egy osztályba de addig is hhh
            if( self.linear_regression_calculation == 'old'):
              _X_center = np.array([before_array[:,2], delta_array[:,0]]).T # center és delta_y (before)
            
            if( self.linear_regression_calculation == 'new'):
              _egy_center   = np.array([before_array[:,2] * before_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              _ketto_center = np.array([before_array[:,2] * delta_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              _X_center_proba = np.stack((_egy_center.flatten(), _ketto_center.flatten()), axis=1)
              _X_center = _X_center_proba
              # nem csak itt kell átírni hanem ahol a lehetséges actionökre is kiszámolja

            _y_center = after_array[:,2].reshape(-1, 1)                   # center (after)
            self.regression_center.fit(_X_center, _y_center)
            self.printer.ba('\t\t ------------------------------- valyon mennyire jó a center metrikának a becslése -----------------------------')
            self.printer.ba('\t\t self.regression_center.coef_ = ', self.regression_center.coef_)
            self.printer.ba('\t\t self.regression_center.intercept_', self.regression_center.intercept_)
            _predicted_center = self.regression_center.predict(_X_center)

            # kiplottolom a before after adatokat egy konkrét szenzor értékeire

  #         plt.scatter(_y_center, _predicted_center)
  #         plt.scatter(before_array[:,2], after_array[:,2], c='black')
            # ezt felváltottam az alábbi három sorral
            # Plot
            # (flag 0 = disable, 1 = plot, 2 = save, 3 = both)
            # Lecseréltem ezt
            if( self.sensor_center_enable == True ):
              if( i % (3 * self.plot_frequency) == 0 ):
                self.plot_before_after_sensor_estimation_in_one_chart(_y_center, _predicted_center, y_delta, 'center', self.plot_before_after_sensor_estimation_flag)
                pass
                # Erre
                # job_for_2A = multiprocessing.Process(target=self.plot_before_after_sensor_estimation_in_one_chart,args=(_y_center, _predicted_center, y_delta, 'center', self.plot_before_after_sensor_estimation_flag))
                # job_for_2A.start()
            
            # [[before_array[:,2](center), after_array[:,2](center), y_delta{action}, time]]
            _array_target_center = np.array([before_array[:,2].ravel(), after_array[:,2].ravel(), y_delta.ravel(), np.arange(0, after_array.shape[0], 1)]).T
            # Lecseréltem ezt
            if( self.sensor_center_enable == True ):
              if( i % (3 * self.plot_frequency) == 0 ):
                self.plot_before_after_sensor_values(_array_target_center, 'center', self.plot_before_after_sensor_values_flag)
                pass
                # Erre
                # job_for_2 = multiprocessing.Process(target=self.plot_before_after_sensor_values,args=(_array_target_center, 'center', self.plot_before_after_sensor_values_flag))
                # job_for_2.start()

  # --------------
  # --------------
  # -------------- right
            self.printer.ba('---------------------------------------')
            self.printer.ba('before_array.shape = ', before_array.shape)
            self.printer.ba('------------------>>>------------------')
            # print(before_array)
            self.printer.ba('------------------<<<------------------')
            # before_array.shape (n_obs, 4)
            # ahol az oszlop a metrika a következő index szerint (3 - right sensor, 2 - center sensor, 1 - left sensor, 0 - self.y)
            _X_right = np.array([before_array[:,3], delta_array[:,0]]).T # right és delta_y (before)
            self.printer.ba(' na akkor ehelyett a képlet helyett    ')
            # Benene van a metrika, hogy éppen most mennyi self.y, mondjuk az is, hogy mennyit akarunk hozzá adni tehát a delta. self.y
            # m′ = c0 + c1 · 80 · 5/(5 + 2) + c2 · 80 · 2/(5 + 2)
            # m′ = c0 + c1 · metrika · self.y/(self.y + delta_y) + c2 · metrika · delta_y/(self.y + delta_y)
            _egy   = np.array([before_array[:,3] * before_array[:,0] / (before_array[:,0] + delta_array[:,0])])
            # _egy = metrika · self.y/(self.y + delta_y)
            # metrika (right) = before_array[:,3]
            # vm              = before_array[:,0]      esetünkben a vm az a self.y egyébként
            # delta_vm        = delta_array[:,0]       esetünkben hogy mennyit változott a self.y a lépés után (vagyis hányat léptünk)
            _ketto   = np.array([before_array[:,3] * delta_array[:,0] / (before_array[:,0] + delta_array[:,0])])
            # _ketto =  metrika · delta_y/(self.y + delta_y)
            # metrika (right) = before_array[:,3]
            # vm              = before_array[:,0]      esetünkben a vm az a self.y egyébként
            # delta_vm        = delta_array[:,0]       esetünkben hogy mennyit változott a self.y a lépés után (vagyis hányat léptünk)
            self.printer.ba('---------------------------------------')
            # A képletet ketté osztotam _egy és _ketto, de nekünk midkettő kell
            # Ebből a kettőből kell csinálnom egy _X_right változót aminek a dimenzió száma (n_obs, 2)
            #
            # Tömbök összefűzése iszonyú szopás
            # aa = np.array([1,2,3,4,5])
            # bb = np.array([6,7,8,9,10])
            # proba1 = np.stack((aa,bb), axis=1)       # <-- ez a jó megoldás
            _X_right_proba = np.stack((_egy.flatten(), _ketto.flatten()), axis=1)
            self.printer.ba('-----------------III-------------------')
            self.printer.ba('Az új _X_right_proba.shape', _X_right_proba.shape)
            self.printer.ba('A régi _X_right.shape     ', _X_right.shape)
            self.printer.ba('-----------------OOO-------------------')
            # Próba képpen akkor most beadom neki az új számítási módszer szerint kéeszült _X_test_proba változót
            # Előtte még kíváncsiságból megnézem ez mennyire más mint ami korábban ment be
            self.printer.ba('>>>>>>>>>>>>>>>>>>> _X_right >>>>>>>>>>>>>>>')
            self.printer.ba(_X_right)
            self.printer.ba('<<<<<<<<<<<<<<<<<<< _X_right_proba <<<<<<<<<')
            self.printer.ba(_X_right_proba)
            self.printer.ba('--------- NA MOST JÖN AZ UGRÁS ÁTVEZETEM AZ ÚJ LR SZÁMÍTÁST --------')
            # Ki akarom majd vezetni az egész számítást egy osztályba de addig is hhh
            if( self.linear_regression_calculation == 'new'):
              _X_right = _X_right_proba

            _y_right = after_array[:,3].reshape(-1, 1)                   # right (after)

            self.regression_right.fit(_X_right, _y_right)
            self.printer.ba('\t\t ------------------------------- valyon mennyire jó a right  metrikának a becslése -----------------------------')
            self.printer.ba('\t\t self.regression_right.coef_ = ', self.regression_right.coef_)
            self.printer.ba('\t\t self.regression_right.intercept_ = ', self.regression_right.intercept_)
            _predicted_right = self.regression_right.predict(_X_right)

            # kiplottolom a before after adatokat egy konkrét szenzor értékeire

  #         plt.scatter(_y_right, _predicted_right)
  #         plt.scatter(before_array[:,3], after_array[:,3], c='black')
            # ezt felváltottam az alábbi három sorral
            # Plot
            # (flag 0 = disable, 1 = plot, 2 = save, 3 = both)
            # Lecseréltem ezt
            if( i % (3 * self.plot_frequency) == 0 ):
              self.plot_before_after_sensor_estimation_in_one_chart(_y_right, _predicted_right, y_delta, 'right', self.plot_before_after_sensor_estimation_flag)
              # Erre
              # job_for_3A = multiprocessing.Process(target=self.plot_before_after_sensor_estimation_in_one_chart,args=(_y_right, _predicted_right, y_delta, 'right', self.plot_before_after_sensor_estimation_flag))
              # job_for_3A.start()
              # [[before_array[:,3](right), after_array[:,3](right), y_delta{action}, time]]
            _array_target_right = np.array([before_array[:,3].ravel(), after_array[:,3].ravel(), y_delta.ravel(), np.arange(0, after_array.shape[0], 1)]).T
            # Lecseréltem ezt
            if( i % (3 * self.plot_frequency) == 0 ):
              self.plot_before_after_sensor_values(_array_target_right, 'right', self.plot_before_after_sensor_values_flag)
              # Erre
              # job_for_3 = multiprocessing.Process(target=self.plot_before_after_sensor_values,args=(_array_target_right, 'right', self.plot_before_after_sensor_values_flag))
              # job_for_3.start()



            # továbbá eszembe jutott az is, hogy nem lenne rossz a left és a rigth szenzor értékének
            # függvényében kimutatni, hogy mi volt a tényleges középponttól vett távolság
            # és azt is, hogy mi volt predicted
            # ez egy háromdimenziós pontfelhő lenne ahol x1->left, x2->rigth z->y_distance, z-> y_distance_predicted

            # Az az igazság, hogy ennek nem a before after részben kéne lennie,
            # de most itt megírom,, utána átteszem egy függvénybe
            # és azt a függvényt akár itt is meghívhatom, de máshol is

            # Az adatok nem a before afterből kellenek nekünk, hanem
            # self.y_distance
            # self.sensor_left
            # self.sensor_center
            # self.sensor_right
            
            # Korábban egybe volt több plotolást is tartalmazott ez a korábbi függvény, amit kivezettem külön fügvényekbe
            # self.plot_state_space_discover(self.plot_state_space_discover_flag)
            
            if( i % (3 * self.plot_frequency) == 0 ):
              # self.plot_state_space_discover_type1(self.plot_state_space_discover_flag)    # right, left, center kapcsolat
              self.plot_state_space_discover_type2(self.plot_state_space_discover_flag)    # right, left, y_distance
              # self.plot_state_space_discover_type3(self.plot_state_space_discover_flag)    # Fehér keret nélkül, right, left, y_distance
              self.plot_state_space_discover_type4(self.plot_state_space_discover_flag)    # 2D, right, left, y_distance
            
            

            # Ez mi ennek utána kéne járni, hogy ez mit csinál ToDo.
            # A test_plot csinálja meg a timeline-t a két szenzorra és a kumulatív hibára
            # Viszont ezt is ToDo kurva szarul írtam meg, egy függvényben 6-7 plottolás is van ezeket ki kell írni külön függvéynekbe
            # ez így teljesen használhatatlan
            if( i % (3 * self.plot_frequency) == 0 ):
              self.plotter.test_plot(self.sensor_left, self.sensor_right, self.y_distance, self.x, self.plotter_flag, self.plotter_switch)

              # self.plotter.test_plot2(self.sensor_left, self.sensor_right, self.y_distance, self.x, self.plotter_flag, self.plotter_switch)
              
              self.plotter.timeline_sensors1(self.sensor_left, self.sensor_right, self.y_distance, self.x, self.plotter_flag)





            # most ki kell számolni, hogy mennyi lenne a szenzorok értéke, ha fel le lépkednénk

            # mondjuk maximalizáljuk a fel le lépkedés mértékét 5-ben

            move = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

            move = np.array([-2, -1, 0, 1, 2])

            move = np.array([-3, -2, -1, 0, 1, 2, 3])

            move = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

            move = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

            # move = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])





            self.printer.action('\t # Az egyes lépések várható kimeneteinek kiszámolása ----------------------------------------------')

            self.printer.action('\t\t # Ennyivel mozdulna el egy szenzor adat 1 egység változással ha 1 lenne a before értéke')
            proba_X_metrika   = np.array([1,1]).reshape(1, -1)
            self.printer.action('proba_X_metrika   = ', proba_X_metrika)
            predicted_proba_left = self.regression_left.predict(proba_X_metrika)
            predicted_proba_center = self.regression_center.predict(proba_X_metrika)
            predicted_proba_right = self.regression_right.predict(proba_X_metrika)

            self.printer.action('-------- 1 y up ->  left   = ', predicted_proba_left)
            self.printer.action('-------- 1 y up ->  center = ', predicted_proba_center)
            self.printer.action('-------- 1 y up ->  right  = ', predicted_proba_right)
            self.printer.action('\n')

            # EGY KURVA NAGY ELMÉLETI DILLEMMÁHOZ ÉRKEZTEM.

            # HÁROM MEGOLDÁS VAN

            # EBBŐL SZERINTEM MOST A LEGROSSZABBAT VÁLASZOTTAM

            # 1, MINDENKÉPPEN VÁLASZT EGYET, A LEGJOBBAT

            # 2, CSAK AKKOR MODOSÍT HA KÖZELEBB TUDJA VINNI A KÖZÉPPONTHOZ MINT AHOL MOST VAN

            # 3, CSAK AKKOR MODOSÍT HA NEM A KÖZÉPPONTON VAN

            # 4, CSAK AKKOR MODOSÍT HA KILÉPETT A SÁVBÓL

            # 5, CSAK AKKOR MODOSÍT HA EGY ELŐRE MEGADOTT ÉRTÉKNÉL JOBBAN ELTÉR A KÖZÉPPONTTÓL
            
            action = 0; tmp = 999999990

            for j in move:
              self.printer.action('\n')
              self.printer.action('\t\t minden j-re kiszámolom a regressziót és be is helyetessítjük a kapott értékekekt a modellbe')
              self.printer.action('\t\t j = ', j)
              # ide be kell helyettesítenem az éppen aktuális értéket, olyan mintha egy új X változót csinálnék amiben csak egy sor van és arra kérnék egy becslést a korábbi modell alapján
              # a bement az éppen aktuális szenzoros érték és az új lépés

              # na majd ezt a bemenetet kell ellenőrizni, hogy stimmel-e a tanítás során használt bemenettel
              _X_left   = np.array([[self.distance_left_from_wall, j]])
              _X_center = np.array([[self.distance_center_from_wall, j]])
              _X_right  = np.array([[self.distance_right_from_wall, j]])

              self.printer.action('\t\t ------------------------ a regresszió bemenetei az éppen aktuális értékek -------------------')
              self.printer.action('\t\t _X_left   = ', _X_left)
              self.printer.action('\t\t _X_center = ', _X_center)
              self.printer.action('\t\t _X_right  = ', _X_right)

              # a fenti értékek valószínűleg jók, de mindíg minden lépésnél ellenőrizni kell

              # hhh
              # teljesen át kell írni a predicted kiszámítását ha igazodni akarok ahhoz a másik képlethez amit korábban már átírtam
              #
              # Pár sorral feljebb állítom elő az _X_right változót amely alapján a predikció elő fog állni
              # Nézzük meg hogy néz ki, mert majd át kell írnom, hogy úgy álljon elő ahogy a korábbi képletnél a fit()-nél kiszámoltam
              # helyzet az, hogy a régi _X_right = sesor_before, delta_vm képlet alapján szerepel
              #       ezzel szemben az új
              # _egy   = np.array([before_array[:,3] * before_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              # _ketto   = np.array([before_array[:,3] * delta_array[:,0] / (before_array[:,0] + delta_array[:,0])])
              # _X_right_proba = np.stack((_egy.flatten(), _ketto.flatten()), axis=1)
              __right  = self.distance_right_from_wall   # before_array[:,3]
              __center = self.distance_center_from_wall  # before_array[:,2]
              __left   = self.distance_left_from_wall    # before_array[:,1]
              __y = self.y                               # before_array[:,0]
              __j = j                                    #  delta_array[:,0]
              __egy_right    = np.array([__right * __y / (__y + __j)])
              __ketto_right  = np.array([__right * __j / (__y + __j)])
              __X_right_new  = np.stack((__egy_right.flatten(), __ketto_right.flatten()), axis=1)

              __egy_center   = np.array([__center * __y / (__y + __j)])
              __ketto_center = np.array([__center * __j / (__y + __j)])
              __X_center_new = np.stack((__egy_center.flatten(), __ketto_center.flatten()), axis=1)

              __egy_left     = np.array([__left * __y / (__y + __j)])
              __ketto_left   = np.array([__left * __j / (__y + __j)])
              __X_left_new   = np.stack((__egy_left.flatten(), __ketto_left.flatten()), axis=1)

              # predicted_left   = self.regression_left.predict(_X_left)           # Old
              # predicted_left   = self.regression_left.predict(__X_left_new)      # New

              # predicted_center = self.regression_center.predict(_X_center)       # Old
              # predicted_center = self.regression_center.predict(__X_center_new)  # New

              # predicted_right  = self.regression_right.predict(_X_right)         # Old
              # predicted_right  = self.regression_right.predict(__X_right_new)    # New

              # Ez de ciki itt mindíg az új alapján számolok akkor is ha model a régi alapján lett feltanítva?

              if( self.linear_regression_calculation == 'old' ):
                predicted_left   = self.regression_left.predict(_X_left)           # Old
                predicted_center = self.regression_center.predict(_X_center)       # Old
                predicted_right  = self.regression_right.predict(_X_right)         # Old
              if (self.linear_regression_calculation == 'new' ):
                predicted_left   = self.regression_left.predict(__X_left_new)      # New
                predicted_center = self.regression_center.predict(__X_center_new)  # New
                predicted_right  = self.regression_right.predict(__X_right_new)    # New


              self.printer.action('\t\t predicted_left   = ', predicted_left)
              self.printer.action('\t\t predicted_center = ', predicted_center)
              self.printer.action('\t\t predicted_right  = ', predicted_right)

              self.printer.action('\t\t --------------------- a regression úgy tűnik, hogy jó és pontos ----------------------')
              self.printer.action('\t\t self.regression_left.coef_   = ', self.regression_left.coef_)
              self.printer.action('\t\t self.regression_center.coef_ = ', self.regression_center.coef_)
              self.printer.action('\t\t self.regression_right.coef_  = ', self.regression_right.coef_)

              self.printer.action('\t\t self.regression_left.intercept_   = ', self.regression_left.intercept_)
              self.printer.action('\t\t self.regression_center.intercept_ = ', self.regression_center.intercept_)
              self.printer.action('\t\t self.regression_right.intercept_  = ', self.regression_right.intercept_)


              # nekünk majd azt az értéket kell választanunk amelyik segítségével a legközelebb jutunk a 0 értékhez

              _X = np.array([predicted_left.ravel(), predicted_center.ravel(), predicted_right.ravel()]).T    # figyelni kell rá, hogy eredetileg is ez volt-e a változók sorrendje

              _X_scaled = self.x_minmaxscaler.transform(_X)

              self.printer.action('\t\t # Ez lesz a bemenete a neurális hálónak')
              self.printer.action('\t\t -------------------------X-------------------------')
              self.printer.action('\t\t ', _X)
              self.printer.action('\t\t -------------------------X_scaled------------------')
              self.printer.action('\t\t ', _X_scaled)
# Elvileg meg lehetne csinálni, hogy az új értékek is mindenféleképen a -1, 1 intervallumba essenek, de jelenleg nem így történik
# Ez nem lesz könnyű

# Ami itt nehéz lesz, hogy a régi X értékekhez, vagy ahhoz amin elvégeztem az x_minmaxscalert hozzá kell csapnom az új linreg által számolt _X
# tömböt és azon megcsiálnom a teljes skálázást (bár ez az egész módszer nem biztos, hogy jó, sőt, de nincs jobb ötletem)
              
              self.printer.action('\t\t ---------------Brutálisan hülye dolgot jelez előre ezért ellenőrizni kell, hogy mi a gond. Esetleg a bemeneti adatok?-----------------')

# Lineáris regresszió
#              predicted_position = self.regression.predict(_X)
#              print('\t\t predicted_position linreg model            = ', predicted_position)
# Lineáris regresszió helyett Neurális hálót használok
              predicted_position_scaled = self.mlp.predict(_X_scaled)
              self.printer.action('\t\t predicted_position neural net model scaled = ', predicted_position_scaled)
# Vissza kell transzformálnom eredeti formájába
              predicted_position = self.y_minmaxscaler.inverse_transform(predicted_position_scaled.reshape(-1, 1))
              self.printer.action('\t\t predicted_position neural net model inverz = ', predicted_position)

              self.printer.action('\t\t --------------------------------------------------------------------------------------------------------------------------------------')

              # legyünk bátrak és módosítsuk az autó self.y pozicióját

              # azzal az értékkel amely abszolút értékben a legkissebb, helyett
              # mivel a célváltozónk akkor jó ha 0, mivel a középvonaltól mért eltérés
              # ezért itt azt az értéket kell kiválasztani ami a legközelebb van 0-hoz

              # természetesen ezen változtatni kell ha nem a középvonaltól való eltérés mértékét akarjuk becsülni
              # de ahhoz fent is át kell állítani hogy mi legyen a self.y_distance számítása

              if( abs(0 - predicted_position) < tmp):       # rossz - javítva - tesztelés alatt
                action = j
                tmp = abs(0 - predicted_position)
                self.printer.action('\t\t ---------------------')
                self.printer.action('\t\t  action = ', action)
                self.printer.action('\t\t  predicted_position = ', predicted_position)
                self.printer.action('\t\t  absolute distance from 0 (tmp) = ', tmp)
                self.printer.action('\t\t ---------------------')

              self.printer.action('\t\t adott j-re {0} kiszámoltuk az előrejelzést de még nem hoztunk döntést -----------------------------------------------------------------'.format(j))
              self.printer.action('\t\t --------------------------------------------------------------------------------------------------------------------------------------')
            
            self.printer.action('\t minden j-re kiszámoltuk az előrejelzést de még nem hoztunk döntést -------------------------------------------------\n')
# igazság szerint ez a kör minden lépésben lefut ha már van elég before after adatunk


# a döntés azonban csak akkor fut le ha az alábbi feltétel teljesül, de igazából korábban már be van ágyazva ugyan ebbe a feltételbe

# version 20. if( i % 3 == 0 ) -> version 22. if( i % 3 == 1 )

          if( i % 3 == 0 ):                      # ugyan ez a feltétel amikor tanítom az út közepének a becslésére

            # gozer ezt csak akkor lépje meg ha az action nem 0
            # mivel ilyenkor is beteszi magát a before after listába és ezzel összezavar mindent
# version 30
# CheckCheck
# Itt lehet átállítani, hogy csak akkor tegye be az értékeket a before after listába ha tényleges van action vagy akkor is ha nincsen
#
# Kicsit belehackeltem és kivezettem egy atributumba ezt a beállítás (self.action_zero_is_allowed = False gyárilag)
            predicate = -123456 if self.action_zero_is_allowed == True else 0

            if( action != predicate ):                # new
            #if( action != 0 ):                       # old
              self.printer.takeaction('------------------------------ IF i % 3 == 0 ------------------------------')
              _summary_action_was_taken = 1
              self.printer.takeaction('=================== TAKE ACTION ===================')
  # ez lett új az ML Auto 10.ipynb-hoz képest
              self.before.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))
              self.printer.takeaction('-------- ennyivel módosítom self.y értékét --------')
              self.printer.takeaction('action = ', action)
              self.printer.takeaction('self.y régi értéke = ', self.y)
              # new v.27
              self.y = self.y + action
              # new v.27 end
  # ez lett új az ML Auto 10.ipynb-hoz képest
              self.calculate_distances()
              self.after.append(np.array([self.y, self.distance_left_from_wall, self.distance_center_from_wall, self.distance_right_from_wall]))

              self.printer.takeaction('self.y új értéke   = ', self.y)
              self.printer.takeaction('action             = ', action)
              self.printer.takeaction('----------------- módosítás vége -----------------')




          # újra kell gondolni az egészet, ugyanis akkor is ki kell számolni a before after értéket amikor modosítom a pozicióját,
          # vagyis végig kell gondolni ezt az egészet.
          # az első elképzelésem az volt, hogy a poziciót csak bizonyos esetben modosíthatom, csak akkor amikor nincs tanítás, és nincs szimulált emelés, vagy csökkentés sem
          # utóbbit az if( i % 3 == 2) feltétellel szűrtem



# Ez a rész itt mindíg lefut

      # new v.25

      # Tároljuk el minden körben a ml modellek érétkeit (lehet, hogy ez egy kicsit lassítani fogja a futás)

      # if hasattr(auto.regression_left, 'coef_'):
      # fffffffffff
      if hasattr(self.regression_left, 'coef_'):
        self.regression_left_coef_history.append(self.regression_left.coef_)
        self.regression_center_coef_history.append(self.regression_center.coef_)
        self.regression_right_coef_history.append(self.regression_right.coef_)

      # new v.25 end
    
      # az mlp veszteség értékét (mlp.loss) is tároljuk el későbbi elemzésre minden körben
      if(hasattr(self.mlp, 'loss_')):
        self.loss_holder.append(self.mlp.loss_)

      # adjuk hozzá az értéket a self.y_history-hoz
      self.y_history.append(self.y)
      if( silent == False ):
        self.printer.util('# A run ciklus vége ------------------------------------------------------------------------------------------------------------------------------------------')
        self.printer.util('#   itt adom hozzás a self.y a self.y_history-hoz')
        self.printer.util('#    self.y :{}'.format(self.y))
        self.printer.util('# \t\t\t --------------- Summary ---------------')
        self.printer.util('# \t\t\t _summary_mlp_fit_was_taken         = ', _summary_mlp_fit_was_taken)
        self.printer.util('# \t\t\t _summary_mlp_prediction_was_taken  = ', _summary_mlp_prediction_was_taken)
        self.printer.util('# \t\t\t _summary_mesterseges_mozgatas      = ', _summary_mesterseges_mozgatas)
        self.printer.util('# \t\t\t _summary_action_were_taken         = ', _summary_action_was_taken)
        self.printer.util('# ')
        self.printer.util('# A run ciklus vége ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
      # Egy nagyon hasznos kiegészítés ha a programot Jupyter Notebookban futtatom
      if ( i % 10 == 0 ):
        clear_output(wait=True)
        pass








# --------------------------------------------------------------------------------------------------







def plot_history(auto, flag, autoscale = True, fileName = 'history'):
  if( flag != 0 ):
    fileName = fileName
    fig, ax = auto.road.show()
    # if( autoscale == True ):
    #   _wall_left_max   = auto.road.wall_left.max()
    #   _wall_left_min   = auto.road.wall_left.min()
    #   _wall_right_max  = auto.road.wall_right.max()
    #   _wall_right_min  = auto.road.wall_right.min()
    #   _wall_center_max = auto.road.wall_center.max()
    #   _wall_center_min = auto.road.wall_center.min()
    #   _top    = np.array([_wall_left_max, _wall_right_max, _wall_center_max]).max()
    #   _bottom = np.array([_wall_left_min, _wall_right_min, _wall_center_min]).min()
    #   ax.set_ylim(_bottom - 10, _top + 10)
    circle = plt.Circle((auto.x, auto.y), 5, color='black')
    ax.add_patch(circle)
    # v.24 - add standardized color -> left = green, rigth = orange
    ax.plot(range(int(auto.x), int(auto.x + auto.distance_center_from_wall)), np.repeat(auto.y, auto.distance_center_from_wall))
    # ax.plot(range(int(self.x), int(self.x+self.distance_left_from_wall)), range(int(self.y), int(self.y+self.distance_left_from_wall)))
    # ax.plot(range(int(self.x), int(self.x+self.distance_right_from_wall)), range(int(self.y), int(self.y-self.distance_right_from_wall), -1))
    print('self.distance_right_from_wall = ', auto.distance_right_from_wall)
    print('self.distance_left_from_wall  = ', auto.distance_left_from_wall)
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_left[auto.x], color='orange')
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_right[auto.x], color='blue')
    if( len(auto.y_history) > 0 ):
      ax.plot(auto.y_history)
      ax.set_title('#i = ' + str(auto.x), fontsize=18, fontweight='bold')
    if( flag == 1 or flag == 3 ): plt.show();
    if( flag == 2 or flag == 3 ): fig.savefig(fileName + '{0:04}'.format(auto.x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');





def plot_history_fixed(auto, flag, ymin, ymax, width, height):
  if( flag != 0 ):
    fig, ax = auto.road.show(width, height)
    circle = plt.Circle((auto.x, auto.y), 5, color='black')
    ax.add_patch(circle)
    # v.24 - add standardized color -> left = green, rigth = orange
    ax.plot(range(int(auto.x), int(auto.x + auto.distance_center_from_wall)), np.repeat(auto.y, auto.distance_center_from_wall))
    # ax.plot(range(int(self.x), int(self.x+self.distance_left_from_wall)), range(int(self.y), int(self.y+self.distance_left_from_wall)))
    # ax.plot(range(int(self.x), int(self.x+self.distance_right_from_wall)), range(int(self.y), int(self.y-self.distance_right_from_wall), -1))
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_left[auto.x])
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_right[auto.x])
    ax.set_ylim([ymin, ymax])
    if( len(auto.y_history) > 0 ):
      ax.plot(auto.y_history)
      ax.set_title('#i = ' + str(auto.x), fontsize=18, fontweight='bold')
    if( flag == 1 or flag == 3 ): plt.show();
    if( flag == 2 or flag == 3 ): fig.savefig('history{0:04}'.format(auto.x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');




def plot_history_range(auto, flag, start, end, autoscale = True):
  if( flag != 0 ):
    fig, ax = auto.road.show()
    start = 0 if start < 0 else start
    end   = len(auto.y_history) if end < start else end
    end   = len(auto.y_history) if end > len(auto.y_history) else end
    if( autoscale == True ):
      _wall_left_max   = auto.road.wall_left[start:end].max()
      _wall_left_min   = auto.road.wall_left[start:end].min()
      _wall_right_max  = auto.road.wall_right[start:end].max()
      _wall_right_min  = auto.road.wall_right[start:end].min()
      _wall_center_max = auto.road.wall_center[start:end].max()
      _wall_center_min = auto.road.wall_center[start:end].min()
      _top    = np.array([_wall_left_max, _wall_right_max, _wall_center_max]).max()
      _bottom = np.array([_wall_left_min, _wall_right_min, _wall_center_min]).min()
      ax.set_ylim(_bottom - 10, _top + 10)
    ax.set_xlim(start, end)
    circle = plt.Circle((auto.x, auto.y), 5, color='black')
    ax.add_patch(circle)
    # v.24 - add standardized color -> left = green, rigth = orange
    ax.plot(range(int(auto.x), int(auto.x+auto.distance_center_from_wall)), np.repeat(auto.y, auto.distance_center_from_wall))
    # ax.plot(range(int(self.x), int(self.x+self.distance_left_from_wall)), range(int(self.y), int(self.y+self.distance_left_from_wall)))
    # ax.plot(range(int(self.x), int(self.x+self.distance_right_from_wall)), range(int(self.y), int(self.y-self.distance_right_from_wall), -1))
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_left[auto.x])
    ax.vlines(x = auto.x, ymin = auto.y, ymax = auto.road.wall_right[auto.x])
    if( len(auto.y_history) > 0 ):
      ax.plot(auto.y_history)
      ax.set_title('#i = ' + str(auto.x), fontsize=18, fontweight='bold')
    if( flag == 1 or flag == 3 ): plt.show();
    if( flag == 2 or flag == 3 ): fig.savefig('history{0:04}'.format(auto.x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');




# Todo: Ez a függvény két plottot is tartalmaz, a második ráadásul eléggé nem szabványos méretű a többihez képest
# Todo: Ezt a másoidik plottot szerintem már kivezettem valahol és semmi keresni valója it ezt majd átnézni
def plot_trace(auto, freq, flag):

  if( flag != 0 ):

    if( auto.x % freq == 0 ):

      if( len(auto.y_history) > 0 ):

        fileName = 'trace'
        fig, ax = auto.road.show()
        circle = plt.Circle((auto.x, auto.y), 5, color='black')
        ax.add_patch(circle)
        ax.plot(range(int(auto.x), int(auto.x + auto.distance_center_from_wall)), np.repeat(auto.y, auto.distance_center_from_wall))
        ax.plot(range(int(auto.x), int(auto.x + auto.distance_left_from_wall)), range(int(auto.y), int(auto.y+auto.distance_left_from_wall)))
        ax.plot(range(int(auto.x), int(auto.x + auto.distance_right_from_wall)), range(int(auto.y), int(auto.y-auto.distance_right_from_wall), -1))
        y_history_array = np.array(auto.y_history)
        y_history_diff = np.diff(y_history_array, n=1, axis=-1, prepend=0)
        y_history_diff[0] = 0
        y_move = np.zeros(auto.road.distance.shape[0])
        y_move[0:y_history_diff.shape[0]] = y_history_diff
        x = np.arange(auto.road.distance.shape[0])
        ax.plot(auto.road.wall_center[0] + y_move * 10)
        ax.plot(auto.y_history)
        plt.title('#i = ' + str(auto.x))
        if( flag == 1 or flag == 3 ): plt.show();
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(auto.x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');

  if( flag != 0 ):

    if( auto.x % freq == 0 ):

      if( len(auto.y_history) > 0 ):

        fileName = 'y_move'
        fig = plt.figure(figsize=(24, 5))
        y_move = np.diff(np.array(auto.y_history), 1, -1, prepend=0)
        y_move[0] = 0
        plt.plot(y_move)
        plt.hlines(0, 0, 100)
        plt.title('#i = ' + str(auto.x))
        # plt.title('#i = ' + str(auto.x), fontsize=18, fontweight='bold');
        if( flag == 1 or flag == 3 ): plt.show();
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(auto.x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');





import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func