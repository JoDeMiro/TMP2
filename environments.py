import numpy as np
import matplotlib.pyplot as plt

class Road():
  def __init__(self, wide, length, type=1, v=124, shift=0, strech=0, noise=0, b=0, cdr = 0.2):
    self.shift       = shift  # 0
    self.strech      = strech # 0
    self.length      = length # 3000
    self.distance    = np.arange(0, self.length, 1)
    self.wide        = wide
    self.wall_right  = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
    self.wall_right[0:100] = 60
    self.wall_left   = self.wall_right + self.wide
    self.wall_center = ( self.wall_left + self.wall_right ) / 2

    # Azt kell megcsinálni, hogy az egyik utat drifteje el, de anélkül, hogy a középpont változna, az legyen a változatlan hagyott uthoz kötve

    if(type == 99):
        def func(x):
            f = 30*(np.sin(x/180)) + x * 0.3 + 30 * np.cos(x/30) + 50 * np.sin(x/90)
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance)
        # A wall_left-et kell eltolni
        self.wall_left   = func(self.distance + self.shift) + self.wide
        # A center pedig nem a kettő átlaga legyen, hanem a self.wall_right + (self.wide / 2)
        # self.wall_center = ( self.wall_left + self.wall_right ) / 2
        self.wall_center = self.wall_right + (self.wide / 2)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]

    # Azt kell megcsinálni, hogy az egyik utat kicsit jobban nyújtsa, vagyis lasabb legyen a ciklusa, az út közepe pedig a másik faltól fix

    if(type == 98):
        def func(x, strech):
            f = 30*(np.sin(x/180 + strech)) + x * 0.3 + 30 * np.cos(x/30) + 50 * np.sin(x/90)
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0)
        # A wall_left-et kell megnyújtani
        self.wall_left   = func(self.distance, self.strech) + self.wide
        # A center pedig nem a kettő átlaga legyen, hanem a self.wall_right + (self.wide / 2)
        # self.wall_center = ( self.wall_left + self.wall_right ) / 2
        self.wall_center = self.wall_right + (self.wide / 2)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]

    # Azt kell megcsinálni, hogy az egyik utat kicsit jobban nyújtsa, vagyis lasabb legyen a ciklusa, az út közepe pedig a másik faltól fix

    if(type == 97):
        def func(x, strech):
            f = 30*(np.sin(x/(180 + strech))) + x * 0.3 + 30 * np.cos(x/30) + 50 * np.sin(x/90)
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0)
        # A wall_left-et kell megnyújtani
        self.wall_left   = func(self.distance, self.strech) + self.wide
        # A center pedig nem a kettő átlaga legyen, hanem a self.wall_right + (self.wide / 2)
        # self.wall_center = ( self.wall_left + self.wall_right ) / 2
        self.wall_center = self.wall_right + (self.wide / 2)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]

    # Azt kell megcsinálni, hogy az egyik utat kicsit jobban nyújtsa, vagyis lasabb legyen a ciklusa, az út közepe pedig a másik faltól fix

    if(type == 96):
        def func(x, strech):
            f = 30*(np.sin(x/(180 + strech))) + x * 0.3 + 30 * np.cos(x/(30 + strech)) + 50 * np.sin(x/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0)
        # A wall_left-et kell megnyújtani
        self.wall_left   = func(self.distance, self.strech) + self.wide
        # A center pedig nem a kettő átlaga legyen, hanem a self.wall_right + (self.wide / 2)
        # self.wall_center = ( self.wall_left + self.wall_right ) / 2
        self.wall_center = self.wall_right + (self.wide / 2)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]


    # 99 - 96
    #
    # Ezek a jó eredmények amelyeket a ML_Auto_V28.ipynb adtak, azért van mert az 'út' egyik 'falától' még determinisztikus volt a távolsága

    # Ezen most változtatunk és az út közepe a két fal közötti távolság fele lesz, ezáltal csak a 99-es beállítás esetén lesz determinisztikus
    # ha a két fal nincs szinkronban akkor nem lesz determinisztikus kapcsolat egyik változó között sem és az út között sem.

    if(type == 95):
        def func(x, shift, strech):
            f = 30*(np.sin((x + shift)/(180 + strech))) + (x + shift) * 0.3 + 30 * np.cos((x + shift)/(30 + strech)) + 50 * np.sin((x + shift)/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0, 0)                                    # őt nem bántjuk
        # A wall_left eltolás és nyújtás is van rajta
        self.wall_left   = func(self.distance, self.shift, self.strech) + self.wide    # a másik falat viszont jól megzavarjuk
        # A center ismét a két fal átlage és nem a self.wall_right + (self.wide / 2)    # Azért, hogy egyik fallal se legyen determinisztikus
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        # self.wall_center = self.wall_right + (self.wide / 2)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]


    # A 95 folytatása azzal a különbséggel, hogy lehet adni neki egy kis zajt, csak a falakra

    if(type == 94):
        def func(x, shift, strech):
            f = 30*(np.sin((x + shift)/(180 + strech))) + (x + shift) * 0.3 + 30 * np.cos((x + shift)/(30 + strech)) + 50 * np.sin((x + shift)/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0, 0)                                    # őt nem bántjuk
        # A wall_left eltolás és nyújtás is van rajta
        self.wall_left   = func(self.distance, self.shift, self.strech) + self.wide    # a másik falat viszont jól megzavarjuk
        # A center ismét a két fal átlage és nem a self.wall_right + (self.wide / 2)    # Azért, hogy egyik fallal se legyen determinisztikus
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        # self.wall_center = self.wall_right + (self.wide / 2)
        # Add Noise
        def noiser(x, noise):
            _tmp = np.random.randn(x.size)*noise
            return x + _tmp
        self.wall_right = noiser(self.wall_right, noise)
        self.wall_left  = noiser(self.wall_left, noise)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]


    # A Ez mind szép és jó, de a target még mindíg determinisztikus a két vonaltól, -> mivel (wall_center = (wall_left + wall_right) /2)

    # Ezen fogok most egy picit változtatni -> lesz a centernek egy saját fluktuációja

    if(type == 89):
        def func(x, shift, strech):
            f = 30*(np.sin((x + shift)/(180 + strech))) + (x + shift) * 0.3 + 30 * np.cos((x + shift)/(30 + strech)) + 50 * np.sin((x + shift)/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0, 0)                                    # őt nem bántjuk
        # A wall_left eltolás és nyújtás is van rajta
        self.wall_left   = func(self.distance, self.shift, self.strech) + self.wide    # a másik falat viszont jól megzavarjuk
        # A center ismét a két fal átlage és nem a self.wall_right + (self.wide / 2)    # Azért, hogy egyik fallal se legyen determinisztikus
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        # self.wall_center = self.wall_right + (self.wide / 2)
        # Add Noise
        def noiser(x, noise):
            _tmp = np.random.randn(x.size)*noise
            return x + _tmp
        self.wall_right = noiser(self.wall_right, noise)
        self.wall_left  = noiser(self.wall_left, noise)
        # Add some non-det function to center
        def linear(x, b):
            _tmp = np.linspace(0, b, x.size)
            return x + _tmp
        self.wall_center = linear(self.wall_center, b)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]


    # Concept drift

    # Egy ideig a ball falhol van közelebb a target -> Aztán átvált és a másik falhoz lesz közelebb

    if(type == 79):
        def func(x, shift, strech):
            f = 200 + 30*(np.sin((x + shift)/(180 + strech)))  + 50 * np.sin((x + shift)/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0, 0)                                    # őt nem bántjuk
        # A wall_left eltolás és nyújtás is van rajta
        self.wall_left   = func(self.distance, self.shift, self.strech) + self.wide    # a másik falat viszont jól megzavarjuk
        # egy ideig az egyik falhoz, majd a másikhoz lesz közelebb
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        def drift(x, rate_point):
            cut_point = int(x.size * rate_point)
            _tmp = np.zeros(x.size)
            print('cut_point ', cut_point)
            for i in range(x.size):
                if( i < cut_point ):
                    _tmp[i] = ( self.wall_left[i] + self.wall_right[i] ) / 2.1
                else:
                    _tmp[i] = ( self.wall_left[i] + self.wall_right[i] ) / 1.9
            return _tmp
        # Add Noise
        def noiser(x, noise):
            _tmp = np.random.randn(x.size)*noise
            return x + _tmp
        self.wall_right = noiser(self.wall_right, noise)
        self.wall_left  = noiser(self.wall_left, noise)
        # Add some drift
        # self.wall_center = drift(self.wall_center, 0.2)
        self.wall_center = drift(self.wall_center, cdr)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]



    # 2 Concept drift nem csak egy

    # Egy ideig a ball falhol van közelebb a target -> Aztán átvált és a másik falhoz lesz közelebb

    if(type == 78):
        def func(x, shift, strech):
            f = 200 + 30*(np.sin((x + shift)/(180 + strech)))  + 50 * np.sin((x + shift)/(90 + strech))
            return f
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_right  = func(self.distance, 0, 0)                                    # őt nem bántjuk
        # A wall_left eltolás és nyújtás is van rajta
        self.wall_left   = func(self.distance, self.shift, self.strech) + self.wide    # a másik falat viszont jól megzavarjuk
        # egy ideig az egyik falhoz, majd a másikhoz lesz közelebb
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        def drift(x, rate_point):
            cut_point_first = int(x.size * rate_point)
            cut_point_second = int(x.size * (1 - rate_point))
            print(cut_point_first)
            print(cut_point_second)
            _tmp = np.zeros(x.size)
            for i in range(x.size):
                if( i < cut_point_first ):
                    _tmp[i] = ( self.wall_left[i] + self.wall_right[i] ) / 2.1
                elif( i >= cut_point_first and i <= cut_point_second ) :
                    _tmp[i] = ( self.wall_left[i] + self.wall_right[i] ) / 1.9
                elif( i > 1 - cut_point_second ):
                    _tmp[i] = ( self.wall_left[i] + self.wall_right[i] ) / 2.1
            return _tmp
        # Add Noise
        def noiser(x, noise):
            _tmp = np.random.randn(x.size)*noise
            return x + _tmp
        self.wall_right = noiser(self.wall_right, noise)
        self.wall_left  = noiser(self.wall_left, noise)
        # Add some drift
        # self.wall_center = drift(self.wall_center, 0.2)
        self.wall_center = drift(self.wall_center, cdr)
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]




    # Random Walk

    # Csak kíváncsiságból, hogy teljesítene egy random walkon (ha mindhárom, a szenzorok és a target is ugya az)

    if(type == 69):
        def randomwalk(x, m):
            a = np.zeros(x.size)
            r = np.random.randn(a.size) * m
            c = np.add.accumulate(r) + 1000
            return c
            
        self.length      = length # 3000
        self.distance    = np.arange(0, self.length, 1)
        self.wall_center = randomwalk(self.distance, 10)                                    # randomwalk
        self.wall_right  = self.wall_center - wide
        self.wall_left   = self.wall_center + wide
        self.wall_right[0:100] = self.wall_right[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_left[0:100] = self.wall_left[101]

    
    if(type == 2):
        # v = 124
        u = 0
        a = 41
        b = 0.3
        c = 30
        d = 30
        e = 50
        f = 90
        self.wall_left   = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
        self.distance += u
        self.wall_right  = a*(np.sin(self.distance/180)) + self.distance * b + c * np.cos(self.distance/d) + e * np.sin(self.distance/f)
        self.wall_right  += v
        self.wall_center = ( 1.3 * self.wall_left + 0.7 * self.wall_right ) / 2
        self.wall_left[0:100]   = self.wall_left[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_right[0:100]  = self.wall_right[101]
        
    if(type == 3):
        v = 224
        u = 0
        a = 41
        b = 0.3
        c = 30
        d = 46
        e = 50
        f = 90
        self.wall_left   = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
        self.distance += u
        self.wall_right  = a*(np.sin(self.distance/180)) + self.distance * b + c * np.cos(self.distance/d) + e * np.sin(self.distance/f)
        self.wall_right  += v
        self.wall_center = ( self.wall_left + self.wall_right ) / 2
        self.wall_left[0:100]   = self.wall_left[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_right[0:100]  = self.wall_right[101]
    
    if(type == 4): # wizu
        # v = 200
        u = 0
        a = 41
        b = 0.3
        c = 30
        d = 43
        e = 50
        f = 90
        n = 2
        self.wall_left   = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
        self.wall_left[0:100] = 60
        self.distance += u
        self.wall_right  = a*(np.sin(self.distance/180)) + self.distance * b + c * np.cos(self.distance/d) + e * np.sin(self.distance/f)
        self.wall_right  += v
        self.wall_center = ( self.wall_left + self.wall_right ) / n
        self.wall_left[0:100]   = self.wall_left[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_right[0:100]  = self.wall_right[101]

    if(type == 5): # wizu
        # v = 200
        u = 0
        a = 41
        b = 0.3
        c = 30
        d = 43
        e = 50
        f = 90
        n = 3
        self.wall_left   = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
        self.wall_left[0:100] = 60
        self.distance += u
        self.wall_right  = a*(np.sin(self.distance/180)) + self.distance * b + c * np.cos(self.distance/d) + e * np.sin(self.distance/f)
        self.wall_right  += v
        self.wall_center = ( self.wall_left + self.wall_right ) / n
        self.wall_left[0:100]   = self.wall_left[101]
        self.wall_center[0:100] = self.wall_center[101]
        self.wall_right[0:100]  = self.wall_right[101]
    
    
    # Be kell állítani a min max értékeket, hogy ehhez igazítsa a plot y_lim értékeit
    self.set_min_max()

    # Description
    self.description()



  def wizu(self, u = 100, v = 100, a=30, b=0.3, c=30, d=30, e=50, f=90, n=2):
    self.wall_left   = 30*(np.sin(self.distance/180)) + self.distance * 0.3 + 30 * np.cos(self.distance/30) + 50 * np.sin(self.distance/90)
    self.wall_left[0:100] = 60
    self.distance += u
    self.wall_right  = a*(np.sin(self.distance/180)) + self.distance * b + c * np.cos(self.distance/d) + e * np.sin(self.distance/f)
    self.wall_right  += v
    self.wall_center = ( self.wall_left + self.wall_right ) / n
    self.wall_center = ( 1.3 * self.wall_left + 0.7 * self.wall_right ) / n
    self.wall_left[0:100]   = self.wall_left[101]
    self.wall_center[0:100] = self.wall_center[101]
    self.wall_right[0:100]  = self.wall_right[101]

    plt.figure(figsize=(20,5)); plt.plot(self.wall_left); plt.plot(self.wall_right); plt.plot(self.wall_center); plt.show()



  def show(self, widht = 26, height = 10):
    fig, ax = plt.subplots(figsize=(widht, height));
    ax.plot(self.wall_left);
    ax.plot(self.wall_right);
    ax.plot(self.wall_center);
    # _y_max = np.max(self.wall_left)
    # ax.set_ylim(40, _y_max);
    ax.set_ylim(self.wall_min - 10, self.wall_max + 10)
    return fig, ax

  
  def description(self):
    print('# ----------------------------------------- road Description -----------------------------------------')
    print('  \t\t road.length = ', self.length)
    print('  \t\t minimum slope (descending) = ', np.min(np.diff(self.wall_center, 1, -1, prepend=self.wall_center[0])))
    print('  \t\t maximum slope (ascending)  =  ', np.max(np.diff(self.wall_center, 1, -1, prepend=self.wall_center[0])))
    print('# ----------------------------------------------------------------------------------------------------')

  def set_min_max(self):
    _wall_left_max = self.wall_left.max()
    _wall_left_min = self.wall_left.min()
    _wall_center_max = self.wall_center.max()
    _wall_center_min = self.wall_center.min()
    _wall_right_max = self.wall_right.max()
    _wall_right_min = self.wall_right.min()
    self.wall_max = np.array([_wall_left_max, _wall_center_max, _wall_right_max]).max()
    self.wall_min = np.array([_wall_left_min, _wall_center_min, _wall_right_min]).min()
