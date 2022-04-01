import matplotlib.pyplot as plt
import numpy as np

import MLPPlot
from MLPPlot import DrawNN


class PostPlotter():
  def __init__(self, car):
    self.car = car

# new v.25
#  def_plot_lr_weights(self, flag):
#    # a lineáris regresszió súlyainak idővonalon való ábrázolása a cél
#    fig = plt.figure(figsize=(10, 6))
#    ax = fig.add_subplot()
#    ax.plot()
#    plt.show()

  def plot_history(self, flag):
    '''
    A PostPlotter osztály inicializálásánál kapott Car objektum
    plot_history(falg) metódusát hívja meg
    '''
    self.car.plot_history(flag)

  def plot_history_self(self):
      fig, ax = self.car.road.show()
      circle = plt.Circle((self.car.x, self.car.y), 5, color='black')
      ax.add_patch(circle)
      # v.24 - add standardized color -> left = green, rigth = orange
      ax.plot(range(int(self.car.x), int(self.car.x+self.car.distance_center_from_wall)), np.repeat(self.car.y, self.car.distance_center_from_wall))
      # print('self.distance_right_from_wall = ', self.distance_right_from_wall)
      # print('self.distance_left_from_wall  = ', self.distance_left_from_wall)
      ax.vlines(x = self.car.x, ymin = self.car.y, ymax = self.car.road.wall_left[self.car.x], color='orange')
      ax.vlines(x = self.car.x, ymin = self.car.y, ymax = self.car.road.wall_right[self.car.x], color='blue')
      if( len(self.car.y_history) > 0 ):
        ax.plot(self.car.y_history)
        ax.set_title('#i = ' + str(self.car.x), fontsize=18, fontweight='bold')
      plt.show()

  def plot_y_distance(self):
    'A Car objektum y_distace atributumát rajzolja ki'
    plt.figure(figsize=(26, 4))
    plt.plot(self.car.y_distance)
    plt.show()

  def plot_y_distance_fix(self):
    'A Car objektum y_distace atributumát rajzolja ki'
    plt.figure(figsize=(26, 4))
    _y_distance = np.zeros(self.car.road.length)
    _end = len(self.car.y_distance)
    _y_distance[0:_end] = np.array(self.car.y_distance)
    plt.plot(_y_distance)
    plt.show()

  def plot_mlp(self):

    num_input_varialbe = ['sensor_left','sensor_center', 'sensor_right']

    # Define the structure of the network
    network_structure = np.hstack(([len(num_input_varialbe)], np.asarray(self.car.mlp.hidden_layer_sizes), [1]))

    print(network_structure)

    # Draw the Neural Network with weights
    network = DrawNN(network_structure, self.car.mlp.coefs_, num_input_varialbe)
    network.draw()

  def plot_y_move_v2(self, car, x, flag, height = 6):

    if( flag != 0 ):

      fileName = 'PostPlotter_y_move_v2'
      fig = plt.figure(figsize=(26, height))
      ax = fig.add_subplot()
      y_move = np.zeros((car.road.length))
      y_move[0:len(car.y_history)] = np.diff(np.array(car.y_history), 1, -1, prepend=0)
      y_move[0] = 0
      y_tick_labels = [-8, -6,-4, -2, '-0.00', 2, 4, 6]
      ax.set_yticklabels(y_tick_labels)
      ax.plot(y_move)
      ax.hlines(0, 0, 100)
      ax.set_title('#i = ' + str(x))
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(x)+'.png', bbox_inches='tight'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');

  def plot_sensors_distibution(self, bins = 30):

    left = np.array(self.car.sensor_left)
    center = np.array(self.car.sensor_center)
    right = np.array(self.car.sensor_right)

    latextext1 = '\n'.join((
    r'$\sigma_{left}  =%.4f$' % (np.std(left)),
    r'$\sigma_{center}=%.4f$' % (np.std(center)),
    r'$\sigma_{right} =%.4f$' % (np.std(right))
    ))

    latextext2 = '\n'.join((
    r'$\overline{x}_{left}=%.4f$' % (np.mean(left)),
    r'$\overline{x}_{center}=%.4f$' % (np.mean(center)),
    r'$\overline{x}_{right}=%.4f$' % (np.mean(right))
    ))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ax = axes[0]
    ax.hist(left, bins=bins, density=True, histtype='step', label='left', color = '#1f77b4')
    ax.hist(center, bins=bins, density=True, histtype='step', label='center', color = '#2ca02c')
    ax.hist(right, bins=bins, density=True, histtype='step', label='right', color = '#ff7f0e')
    ax.legend(loc='upper right')
    ax.text(0.05, 0.81, latextext1, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    ax.set_ylabel('Density')

    ax = axes[1]
    ax.hist(left, bins=bins, density=True, histtype='step', label='left', color='#1f77b4')
    ax.hist(right, bins=bins, density=True, histtype='step', label='right', color='#ff7f0e')
    ax.legend(loc='upper right')
    ax.text(0.05, 0.81, latextext2, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    # ax.set_ylabel('Density')

    fig.suptitle('Distribution of the values of the sensors')
    fig.text(0.5, -0.0, '$\max_{x \in [a,b]}f(x)$', ha='center')
    plt.show()

  def plot_mlp_surface_prediction_v2(self, resolution = 10):
    # fogja meg az auto left és rigth sensor értékeit
    # vegye a minimumot és a maximumát külön külön
    # csináljon rajtuk egy np.linspace-t
    sl = np.array(self.car.sensor_left); sl_min = sl.min(); sl_max = sl.max()
    print('sl.shape = ', sl.shape); print('sl.min() = ', sl_min); print('sl.max() = ', sl_max)
    sr = np.array(self.car.sensor_right); sr_min = sr.min(); sr_max = sr.max()
    print('sr.shape = ', sr.shape); print('sr.min() = ', sr_min); print('sr.max() = ', sr_max)
    sc = np.array(self.car.sensor_center); sc_min = sc.min(); sc_max = sc.max()
    print('sc.shape = ', sc.shape); print('sc.min() = ', sc_min); print('sc.max() = ', sc_max)

    sensor_center = 100
    _sl = np.linspace(sl_min, sl_max, num = resolution)
    _sr = np.linspace(sr_min, sr_max, num = resolution)
    _sc = np.linspace(sensor_center, sensor_center, num = resolution)

    # kell csinálni egy mesh gridet a plothoz
    _left, _right = np.meshgrid(_sl, _sr)

    print('_left.shape   = ', _left.shape); print('_right.shapes = ', _right.shape)

    # a bemeneti vectorhoz -> itt mátrixhoz -> kell csinálnom néhány átalakítást

    _left_input = _left.flatten()
    _right_input = _right.flatten()
    _center_input = np.full((resolution, resolution), sensor_center).flatten()

    # create an input vector
    _X_input = np.array([_left_input, _center_input, _right_input]).T

    # normlaize it
    _X_input_scaled = self.car.x_minmaxscaler.transform(_X_input)

    # predict
    _Y_output_predicted = self.car.mlp.predict(_X_input_scaled)

    # transform
    _Y_predicted_inverse = self.car.y_minmaxscaler.inverse_transform(_Y_output_predicted.reshape(1, -1))

    # vissza kell alakítanom mátrix formába
    _Y_predicted = _Y_predicted_inverse.reshape(resolution, resolution)

    plt.contourf(_left, _right, _Y_predicted, levels = 30)
    plt.colorbar(label='level')
    plt.show()

  # valamiért ez a kettő
  # ha kicsit is de eltérő eredményt ad
  def plot_mlp_surface_prediction_v1(self, resolution = 10):
    # fogja meg az auto left és rigth sensor értékeit
    # vegye a minimumot és a maximumát külön külön
    # csináljon rajtuk egy np.linspace-t
    sl = np.array(self.car.sensor_left); sl_min = sl.min(); sl_max = sl.max()
    print('type(sl) = ', type(sl))
    print('sl.shape = ', sl.shape)
    print('sl.size  = ', sl.size)
    print('sl.min() = ', sl_min)
    print('sl.max() = ', sl_max)
    sr = np.array(self.car.sensor_right); sr_min = sr.min(); sr_max = sr.max()
    print('sr.shape = ', sr.shape)
    print('sr.size  = ', sr.size)
    print('sr.min() = ', sr_min)
    print('sr.max() = ', sr_max)
    sc = np.array(self.car.sensor_center); sc_min = sc.min(); sc_max = sc.max()
    print('sc.shape = ', sc.shape)
    print('sc.size  = ', sc.size)
    print('sc.min() = ', sc_min)
    print('sc.max() = ', sc_max)

    set_sensor_center = 100
    _sl = np.linspace(sl_min, sl_max, num = resolution)
    _sr = np.linspace(sr_min, sr_max, num = resolution)
    _sc = np.linspace(set_sensor_center, set_sensor_center, num = resolution)

    # kell csinálni egy mesh gridet
    _x, _y = np.meshgrid(_sl, _sr)

    print('_x.shape  = ', _x.shape)
    print('_y.shapes = ', _y.shape)

    # az iterációnál vigyezni kell, mert _x és _y rohadtul nem egész számok
    # ugyan ez de most for loop-al csináltam meg
    _z = np.zeros((resolution,resolution))

    for i in range(1, resolution):
      for j in range(1, resolution):
        _left = _x[i][j]
        _right = _y[i][j]
        _center = set_sensor_center
        # meg kell csinálni a prediction ami nem lesz könnyű mert több lépésből áll
        # 1.
        # rakjuk össze a beneti vectort
        _X_input = np.array([_left, _center, _right])
        # print(_X_input)

        # 2.
        # normalizáljuk
        _X_input_scaled = self.car.x_minmaxscaler.transform(_X_input.reshape(1, -1))

        # 3.
        # becsüljünk
        _Y_output_predicted = self.car.mlp.predict(_X_input_scaled)

        # 4.
        # transformáljuk vissza a becsült értékeket
        _Y_predicted_inverse = self.car.y_minmaxscaler.inverse_transform(_Y_output_predicted.reshape(-1, 1))

        # 3.
        # egyébként rájöttem, hogy ezt nem így egyenként kéne megcsinálnom,
        # megcsinálhatnám úgy is, hogy az egészet egyben állítom elő
        # tehát nem lenne szükség erre a nested for loop ciklusra
        _z[i][j] = _Y_predicted_inverse

    plt.contourf(_x, _y, _z, levels = 30)
    plt.colorbar(label='level')
    plt.show()

  def plot_mlp_surface_prediction_v3(self, flag = 1, resolution = 10, transparency = 1, cmap = 'viridis', elevation = 20, azimuth = -35, i = 1):
    # fogja meg az auto left és rigth sensor értékeit vegye a minimumot és a maximumát külön külön
    # csináljon rajtuk egy np.linspace-t
    sl = np.array(self.car.sensor_left); sl_min = sl.min(); sl_max = sl.max()
    # print('sl.shape = ', sl.shape); print('sl.min() = ', sl_min); print('sl.max() = ', sl_max)
    sr = np.array(self.car.sensor_right); sr_min = sr.min(); sr_max = sr.max()
    # print('sr.shape = ', sr.shape); print('sr.min() = ', sr_min); print('sr.max() = ', sr_max)
    sc = np.array(self.car.sensor_center); sc_min = sc.min(); sc_max = sc.max()
    # print('sc.shape = ', sc.shape); print('sc.min() = ', sc_min); print('sc.max() = ', sc_max)

    sensor_center = 100
    _sl = np.linspace(sl_min, sl_max, num = resolution)
    _sr = np.linspace(sr_min, sr_max, num = resolution)
    _sc = np.linspace(sensor_center, sensor_center, num = resolution)

    # kell csinálni egy mesh gridet a plothoz
    _left, _right = np.meshgrid(_sl, _sr)

    print('_left.shape   = ', _left.shape); print('_right.shapes = ', _right.shape)

    # a bemeneti vectorhoz -> itt mátrixhoz -> kell csinálnom néhány átalakítást
    _left_input = _left.flatten()
    _right_input = _right.flatten()
    _center_input = np.full((resolution, resolution), sensor_center).flatten()

    # create an input vector
    _X_input = np.array([_left_input, _center_input, _right_input]).T

    # normlaize it
    _X_input_scaled = self.car.x_minmaxscaler.transform(_X_input)

    # predict
    _Y_output_predicted = self.car.mlp.predict(_X_input_scaled)

    # transform
    _Y_predicted_inverse = self.car.y_minmaxscaler.inverse_transform(_Y_output_predicted.reshape(1, -1))

    # vissza kell alakítanom mátrix formába
    _Y_predicted = _Y_predicted_inverse.reshape(resolution, resolution)

    fileName = 'PostPlotter_3D_MLP_Prediction_'
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection = '3d')
    ax.view_init(elev = elevation, azim = azimuth)  
    # x axist direction ascending descending
    # ax.invert_xaxis()
    # labels
    ax.set_xlabel('sensor left')
    ax.set_ylabel('sensor right')
    ax.set_zlabel('y_distance')
    # color
    szin = np.arange(len(self.car.sensor_right))
    # scatter
    scat = ax.scatter(self.car.sensor_left, self.car.sensor_right, self.car.y_distance, c=szin)
    # surface
    surf = ax.plot_surface(_left, _right, _Y_predicted, cmap=cmap, alpha = transparency)
    # wireframe
    wire = ax.plot_wireframe(_left, _right, _Y_predicted, rstride=20, cstride=20)
    # colorbar -> scatter
    # fig.colorbar(surf, label='level')

    
    if( flag == 1 or flag == 3 ): plt.show()
    if( flag == 2 or flag == 3 ):
      fig.savefig(fileName + '_3D_v1_{0:04}'.format(i)+'.png'); plt.close(fig);
      plt.close('all'); fig.clf(); ax.cla(); plt.close('all');


  def plot_mlp_surface_prediction_v4(self, flag = 1, limit = False, resolution = 10, transparency = 1, cmap = 'viridis', elevation = 20, azimuth = -35, center = 100, i = 1):
    
    if ( flag != 0 ):
    
      # fogja meg az auto left és rigth sensor értékeit vegye a minimumot és a maximumát külön külön
      # csináljon rajtuk egy np.linspace-t
      sl = np.array(self.car.sensor_left); sl_min = sl.min(); sl_max = sl.max()
      # print('sl.shape = ', sl.shape); print('sl.min() = ', sl_min); print('sl.max() = ', sl_max)
      sr = np.array(self.car.sensor_right); sr_min = sr.min(); sr_max = sr.max()
      # print('sr.shape = ', sr.shape); print('sr.min() = ', sr_min); print('sr.max() = ', sr_max)
      sc = np.array(self.car.sensor_center); sc_min = sc.min(); sc_max = sc.max()
      # print('sc.shape = ', sc.shape); print('sc.min() = ', sc_min); print('sc.max() = ', sc_max)

      if ( limit == True ):
        __x_min = 0; __x_max = 200;
        __y_min = 0; __y_max = 200;
        __z_min = -50; __z_max = 50;
        sl_min = __x_min; sl_max = __x_max
        sr_min = __y_min; sr_max = __y_max
        sc_min = sc.min(); sc_max = sc.max()

      sensor_center = center
      _sl = np.linspace(sl_min, sl_max, num = resolution)
      _sr = np.linspace(sr_min, sr_max, num = resolution)
      _sc = np.linspace(sensor_center, sensor_center, num = resolution)

      # kell csinálni egy mesh gridet a plothoz
      _left, _right = np.meshgrid(_sl, _sr)

      # print('_left.shape   = ', _left.shape); print('_right.shapes = ', _right.shape)

      # a bemeneti vectorhoz -> itt mátrixhoz -> kell csinálnom néhány átalakítást
      _left_input = _left.flatten()
      _right_input = _right.flatten()
      _center_input = np.full((resolution, resolution), sensor_center).flatten()

      # create an input vector
      _X_input = np.array([_left_input, _center_input, _right_input]).T

      # normlaize it
      _X_input_scaled = self.car.x_minmaxscaler.transform(_X_input)

      # predict
      _Y_output_predicted = self.car.mlp.predict(_X_input_scaled)

      # transform
      _Y_predicted_inverse = self.car.y_minmaxscaler.inverse_transform(_Y_output_predicted.reshape(1, -1))

      # vissza kell alakítanom mátrix formába
      _Y_predicted = _Y_predicted_inverse.reshape(resolution, resolution)

      fileName = 'PostPlotter_3D_MLP_Prediction_'
      fig = plt.figure(figsize=(10,10))
      ax = fig.add_subplot(projection = '3d')
      ax.view_init(elev = elevation, azim = azimuth)  
      # x axist direction ascending descending
      # ax.invert_xaxis()
      # labels
      ax.set_xlabel('sensor left')
      ax.set_ylabel('sensor right')
      ax.set_zlabel('y_distance')
      # limit
      if (limit == True ):
        ax.set_xlim(__x_min, __x_max)
        ax.set_ylim(__y_min, __y_max)
        ax.set_zlim(__z_min, __z_max)
      # color
      szin = np.arange(len(self.car.sensor_right))
      # scatter
      scat = ax.scatter(self.car.sensor_left, self.car.sensor_right, self.car.y_distance, c=szin)
      # surface
      surf = ax.plot_surface(_left, _right, _Y_predicted, cmap=cmap, alpha = transparency)
      # wireframe
      wire = ax.plot_wireframe(_left, _right, _Y_predicted, rstride=20, cstride=20)

      # contour
      ax.contour3D(_left, _right, _Y_predicted, 70)
      # colorbar -> scatter
      # fig.colorbar(surf, label='level')
      
      if( flag == 1 or flag == 3 ): plt.show()
      if( flag == 2 or flag == 3 ):
        fig.savefig(fileName + '_3D_v1_{0:04}'.format(i)+'.png'); plt.close(fig);
        plt.close('all'); fig.clf(); ax.cla(); plt.close('all');



  # plotter plot_lr_weight
  def plot_lr_weight(self, car, sensors = ['left'], coefs = ['sensor', 'action'], x = 1, flag = 1):

    print(len(car.regression_center_coef_history))
    # print(car.regression_center_coef_history[100][0])
    # print(car.regression_center_coef_history[100][0][0])
    # print(car.regression_center_coef_history[100][0][1])
    # print(car.regression_left.coef_)

    if( flag != 0 ):

          # _X_left   =  [[51 -3]]
          # _X_center =  [[110  -3]]
          # _X_right  =  [[51 -3]]

          # Emlékeztetőül,
          # A regressziók ilyen bemeneteket várnak, tehát az első coefficiens azt mondja meg,
          # hogy adott szezor értéket ekkora súllyal kell figyelmbe venni
          # a második coefficiens pedig azt modja meg, hogy a változtatás irányát ekkora
          # sullyal kell figelembe venni -> ha azt akarjuk meghatározni, hogy adott szezorértékből
          # mi lesz, ha valamennyivel elmozdítjuk az autót.
          # A szezor értéke és ez elmozdítás mértéke változók (paraméterek) az egyenletben
          # a szorzótényezők (az egyenlet coefficiensei pedig állandók)
          # Ez a plott a coefficienseket jelenítit meg, illetve azt, hogy ezek hogyan változtak
          # a futás során.

      # sajnos át kell alakítanom másképpen nem megy

      array_regression_left_coef_history = np.array(car.regression_left_coef_history)
      array_regression_left_coef_history = array_regression_left_coef_history[:,[0][0]]

      array_regression_center_coef_history = np.array(car.regression_center_coef_history)
      array_regression_center_coef_history = array_regression_center_coef_history[:,[0][0]]

      array_regression_right_coef_history = np.array(car.regression_right_coef_history)
      array_regression_right_coef_history = array_regression_right_coef_history[:,[0][0]]

      a = True if 'action' in coefs else False
      s = True if 'sensor' in coefs else False

      fig = plt.figure(figsize=(10, 6))
      ax = fig.add_subplot()

      if ( 'left' in sensors ):
        if a : ax.plot(array_regression_left_coef_history[:,0], c = '#5195c4', linestyle='dashed', label = 'left sensor coef')
        if s : ax.plot(array_regression_left_coef_history[:,1], c = '#5195c4', label = 'left action coef')

      if ( 'center' in sensors ):
        if a : ax.plot(array_regression_center_coef_history[:,0], c = '#000000', linestyle='dashed', label = 'center sensor coef')
        if s : ax.plot(array_regression_center_coef_history[:,1], c = '#000000', label = 'center action coef')

      if ( 'right' in sensors ):
        if a : ax.plot(array_regression_right_coef_history[:,0], c = '#ff8821', linestyle='dashed', label = 'right sensor coef')
        if s : ax.plot(array_regression_right_coef_history[:,1], c = '#ff8821', label = 'right action coef')
      ax.legend(frameon=False)
      # fig.show()

      fileName = 'plot_lr_coefs'
      if( flag == 1 or flag == 3 ): plt.show(); # fig.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(x)+'.png', bbox_inches='tight'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');










# -------------------------------------------------------------------------------


class Plotter():
  def __init__(self):
    pass

  def plot_mlp(self, mlp, flag):

    if( flag != 0 ):

      num_input_varialbe = ['sensor_left','sensor_center', 'sensor_right']

      # Define the structure of the network
      network_structure = np.hstack(([len(num_input_varialbe)], np.asarray(mlp.hidden_layer_sizes), [1]))

      print(network_structure)

      # Draw the Neural Network with weights
      network = DrawNN(network_structure, mlp.coefs_, num_input_varialbe)
      network.draw(flag)
  
  def plot_y_move(self, y_history, x, flag):

    if( flag != 0 ):

      fileName = 'y_move'
      fig = plt.figure(figsize=(10.5, 6))
      y_move = np.diff(np.array(y_history), 1, -1, prepend=0)
      y_move[0] = 0
      plt.plot(y_move)
      plt.hlines(0, 0, 100)
      plt.title('#i = ' + str(x))
      # plt.title('#i = ' + str(x), fontsize=18, fontweight='bold');
      if( flag == 1 or flag == 3 ): plt.show();
      if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_{0:04}'.format(x)+'.png'); plt.close('all'); fig.clf(); ax.cla(); plt.close('all');

  # Legacy - mert sok helyen még ezen a néven használom de átteszem egy másik értelmesebb névre is.
  def test_plot2(self, sensor_left, sensor_right, y_distance, x, flag, lists):

    if( flag != 0 ):

      if( 6 in lists or 99 in lists ):

        fileName = 'timeline_sensors'

        fig, ax1 = plt.subplots(figsize=(15,5))

        ax1.set_title('#i = ' + str(x), fontsize=18, fontweight='bold');

        ax2 = ax1.twinx()

        ax1.plot(sensor_left, label='left distance')
        ax1.plot(sensor_right, label='right distance')
        ax1.plot(y_distance, label='dist. from center')

        err = np.cumsum(np.abs(y_distance))
        ax2.plot(err, c='black', label='cummulative error')

        ax1.set_xlabel('time')
        ax1.set_ylabel('sensor values', color='black')
        ax2.set_ylabel('cummulative error', color='black')

        ax1.legend(frameon=False)
        ax2.legend(frameon=False)

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_v1_{0:04}'.format(x)+'.png'); plt.close(fig)

  # Ezt régen test_plot2 néven volt elérhető
  def timeline_sensors1(self, sensor_left, sensor_right, y_distance, x, flag):

    if( flag != 0 ):

        fileName = 'timeline_sensors'

        fig, ax1 = plt.subplots(figsize=(15,5))

        ax1.set_title('#i = ' + str(x), fontsize=18, fontweight='bold');

        ax2 = ax1.twinx()

        ax1.plot(sensor_left, label='left distance')
        ax1.plot(sensor_right, label='right distance')
        ax1.plot(y_distance, label='dist. from center')

        err = np.cumsum(np.abs(y_distance))
        ax2.plot(err, c='black', label='cummulative error')

        ax1.set_xlabel('time')
        ax1.set_ylabel('sensor values', color='black')
        ax2.set_ylabel('cummulative error', color='black')

        ax1.legend(frameon=False)
        ax2.legend(frameon=False)

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_v1_{0:04}'.format(x)+'.png'); plt.close(fig)



  def test_plot(self, sensor_left, sensor_right, y_distance, x, flag, lists = [0]):

    if( flag != 0 ):

      __x_max = np.max(sensor_left); __x_min = np.min(sensor_left);
      __y_max = np.max(sensor_right); __y_min = np.min(sensor_right);
      __z_max = np.max(y_distance); __z_min = np.min(y_distance);

      __x = sensor_left[-1]; __y = sensor_right[-1]; __z = y_distance[-1];

      fileName = 'state_space_discover_new_plotter'

      if( 1 in lists or 99 in lists ):

        # version 1

        szin = np.arange(len(sensor_right))
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sensor_left, sensor_right, y_distance, c=szin)
        ax.set_xlabel('sensor left')
        ax.set_ylabel('sensor right')
        ax.set_zlabel('y_distance')

        # ax.invert_xaxis()

        ax.set_xlim(__x_min, __x_max);
        ax.set_ylim(__y_min, __y_max);
        ax.set_zlim(__z_min, __z_max);

        xe = 0; xv = 10;            xe = __x_min; xv = __x_max;
        ye = 10; yv = 10;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 10; xv = 10;           xe = __x_min; xv = __x_min;
        ye = 0; yv = 100;           ye = __y_min; yv = __y_max;
        ze = 20; ze = 20;           ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_v1_{0:04}'.format(x)+'.png'); plt.close(fig)




      if( 2 in lists or 99 in lists ):

        # version 2

        szin = np.arange(len(sensor_right))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sensor_left, sensor_right, y_distance, c=szin)
        ax.set_xlabel('sensor left')
        ax.set_ylabel('sensor right')
        ax.set_zlabel('y_distance')

        ax.set_xlim(__x_min, __x_max);
        ax.set_ylim(__y_min, __y_max);
        ax.set_zlim(__z_min, __z_max);

        xe = 0; xv = 10;            xe = __x_min; xv = __x_max;
        ye = 10; yv = 10;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 10; xv = 10;           xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y_min; yv = __y_max;
        ze = 20; ze = 20;           ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')
        
        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_v2_{0:04}'.format(x)+'.png'); plt.close(fig)




      if( 3 in lists or 99 in lists ):

        # version 3

        szin = np.arange(len(sensor_right))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sensor_left, sensor_right, y_distance, c=szin)
        ax.set_xlabel('sensor left')
        ax.set_ylabel('sensor right')
        ax.set_zlabel('y_distance')

        ax.set_xlim(__x_min, __x_max);
        ax.set_ylim(__y_min, __y_max);
        ax.set_zlim(__z_min, __z_max);

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 0; xv = 10;            xe = __x_min; xv = __x_max;
        ye = 10; yv = 10;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 10; xv = 10;           xe = ax.get_xlim()[0]; xv = ax.get_xlim()[0];
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = 20; ze = 20;           ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 10; xv = 10;           xe = __x_min; xv = __x_min;
        ye = 0; yv = 100;           ye = __y_min; yv = __y_max;
        ze = 20; ze = 20;           ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 0; xv = 0;             xe = __x_min; xv = __x_max;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        ax.view_init(elev=20., azim=-35)

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_v3_{0:04}'.format(x)+'.png'); plt.close(fig)



      if( 4 in lists or 99 in lists ):

        # version 4

        szin = np.arange(len(sensor_right))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sensor_left, sensor_right, y_distance, c=szin)
        ax.set_xlabel('sensor left')
        ax.set_ylabel('sensor right')
        ax.set_zlabel('y_distance')

        ax.set_xlim(__x_min, __x_max);
        ax.set_ylim(__y_min, __y_max);
        ax.set_zlim(__z_min, __z_max);

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 0; xv = 10;            xe = __x_min; xv = __x_max;
        ye = 10; yv = 10;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 10; xv = 10;           xe = ax.get_xlim()[0]; xv = ax.get_xlim()[0];
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = 20; ze = 20;           ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 10; xv = 10;           xe = __x_min; xv = __x_min;
        ye = 0; yv = 100;           ye = __y_min; yv = __y_max;
        ze = 20; ze = 20;           ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 0; xv = 0;             xe = __x_min; xv = __x_max;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        # ---
        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange', linestyle='dashed') #dotted

        xe = 0; xv = 0;             xe = __x_min; xv = __x;
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='green', linestyle='dashed') #dotted

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z_min; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue', linestyle='dashed') #dotted

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_v4_{0:04}'.format(x)+'.png'); plt.close(fig)



      if( 5 in lists or 99 in lists ):

        # version 5

        szin = np.arange(len(sensor_right))
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sensor_left, sensor_right, y_distance, c=szin)
        ax.set_xlabel('sensor left')
        ax.set_ylabel('sensor right')
        ax.set_zlabel('y_distance')

        __x_min = 0; __x_max = 200;
        __y_min = 0; __y_max = 200;
        __z_min = -50; __z_max = 50;

        ax.set_xlim(__x_min, __x_max);
        ax.set_ylim(__y_min, __y_max);
        ax.set_zlim(__z_min, __z_max);

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 0; xv = 10;            xe = __x_min; xv = __x_max;
        ye = 10; yv = 10;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = ax.get_zlim()[0]; zv = ax.get_zlim()[0];

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue')

        xe = 10; xv = 10;           xe = ax.get_xlim()[0]; xv = ax.get_xlim()[0];
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = 20; ze = 20;           ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 10; xv = 10;           xe = __x_min; xv = __x_min;
        ye = 0; yv = 100;           ye = __y_min; yv = __y_max;
        ze = 20; ze = 20;           ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv], [ze,zv], c='green')

        xe = 0; xv = 0;             xe = __x_min; xv = __x_max;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y_max; yv = __y_max;
        ze = -50; zv = -50;         ze = __z_min; zv = __z_max;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange')

        # ---
        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = ax.get_ylim()[0]; yv = ax.get_ylim()[1];
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='orange', linestyle='dashed') #dotted

        xe = 0; xv = 0;             xe = __x_min; xv = __x;
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='green', linestyle='dashed') #dotted

        xe = 0; xv = 0;             xe = __x; xv = __x;
        ye = 0; yv = 100;           ye = __y; yv = __y;
        ze = -50; zv = -50;         ze = __z_min; zv = __z;

        ax.plot([xe,xv],[ye,yv],[ze,zv], c='blue', linestyle='dashed') #dotted

        ax.view_init(elev=20., azim=-35)

        if( flag == 1 or flag == 3 ): plt.show()
        if( flag == 2 or flag == 3 ): fig.savefig(fileName + '_LeftRightYDistance_3D_v5_{0:04}'.format(x)+'.png'); plt.close(fig)




  