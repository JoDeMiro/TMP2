class Printer():
  def __init__(self):
    self._nn = True               # Neural Network
    self._lr = False              # Linear Regression on Before After
    self._sr = False              # Sensor Data
    self._ba = False              # Before After Data
    self._nf = False              # Print Info
    self._db = False              # Print Debug
    self._er = True               # Print Error
    self._ut = True               # Print Util
    self._bs = True               # Print Basic
    self._in = True               # Print Investigation
    self._ac = False              # Print Action results
    self._dc = True               # Print Decision result
    self._ta = False              # Print Take Action
    

  def nn(self, text, value = ""):
    if( self._nn == True ):
      print(text, value)

  def lr(self, text, value = ""):
    if( self._lr == True ):
      print(text, value)

  def sr(self, text, value = ""):
    if( self._sr == True ):
      print(text, value)
  
  def ba(self, text, value = ""):
    if( self._ba == True ):
      print(text, value)

  def info(self, text, value = ""):
    if( self._nf == True ):
      print(text, value)
  
  def debug(self, text, value = ""):
    if( self._db == True ):
      print(text, value)
  
  def error(self, text, value = ""):
    if( self._er == True ):
      print(text, value)

  def util(self, text, value = ""):
    if( self._ut == True ):
      print(text, value)

  def basic(self, text, value = ""):
    if( self._bs == True ):
      print(text, value)

  def investigation(self, text, value = ""):
    if( self._in == True ):
      print(text, value)

  def action(self, text, value = ""):
    if( self._ac == True ):
      print(text, value)

  def decision(self, text, value = ""):
    if( self._dc == True ):
      print(text, value)

  def takeaction(self, text, value = ""):
    if( self._ta == True ):
      print(text, value)

  def nn_(self, text):
    if( self._nn == True ):
      print(text)

  def lr_(self, text):
    if( self._lr == True ):
      print(text)

  def sr_(self, text):
    if( self._sr == True ):
      print(text)

  def ba_(self, text):
    if( self._ba == True ):
      print(text)

  def info_(self, text):
    if( self._nf == True ):
      print(text)

  def debug_(self, text):
    if( self._db == True ):
      print(text)
  
  def error_(self, text):
    if( self._er == True ):
      print(text)

  def util_(self, text):
    if( self._ut == True ):
      print(text)
  
  def basic_(self, text):
    if( self._bs == True ):
      print(text)
  
  def investigation_(self, text):
    if( self._in == True ):
      print(text)

  def action_(self, text):
    if( self._ac == True ):
      print(text)

  def decision_(self, text):
    if( self._dc == True ):
      print(text)

  def takeaction_(self, text):
    if( self._ta == True ):
      print(text)

  def __str__(self):
        return " _nn: {0}\n _lr: {1}\n _sr: {2}\n _ba: {3}\n _nf: {4}\n _db: {5}\n _er: {6}\n _ut: {7}\n _bs: {8}\n _in: {9}\n _ac: {10}\n _dc: {11}\n _ta: {12}".format(self._nn, self._lr, self._sr, self._ba, self._nf, self._db, self._er, self._ut, self._bs, self._in, self._ac, self._dc, self._ta)



