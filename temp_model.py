#coding=utf-8

try:
    import __future__
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import os
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import datetime as dt
except:
    pass

try:
    import plotly.express as px
except:
    pass

try:
    import calendar
except:
    pass

try:
    import altair as alt
except:
    pass

try:
    from functools import reduce
except:
    pass

try:
    from datetime import timedelta
except:
    pass

try:
    from dateutil.relativedelta import *
except:
    pass

try:
    from datetime import date
except:
    pass

try:
    from sklearn.cluster import DBSCAN
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import calendar
except:
    pass

try:
    from datetime import timedelta
except:
    pass

try:
    from dateutil.relativedelta import *
except:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
except:
    pass

try:
    from boruta import BorutaPy
except:
    pass

try:
    from sklearn.feature_selection import RFECV
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass

try:
    import seaborn as sns
except:
    pass

try:
    from sklearn.preprocessing import MinMaxScaler
except:
    pass

try:
    from tqdm import tqdm_notebook, tqdm
except:
    pass

try:
    import warnings
except:
    pass

try:
    import time
except:
    pass

try:
    import ipyparallel as ipp
except:
    pass

try:
    from sklearn.metrics import r2_score
except:
    pass

try:
    from sklearn.metrics import mean_absolute_error
except:
    pass

try:
    from sklearn.metrics import mean_squared_error
except:
    pass

try:
    from sklearn.tree import DecisionTreeRegressor
except:
    pass

try:
    from sklearn.linear_model import LinearRegression
except:
    pass

try:
    from sklearn.svm import SVR
except:
    pass

try:
    from sklearn.preprocessing import PolynomialFeatures
except:
    pass

try:
    from sklearn.linear_model import Ridge
except:
    pass

try:
    from sklearn.preprocessing import StandardScaler
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.layers import Dense
except:
    pass

try:
    from keras.callbacks import EarlyStopping
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from sklearn.tree import DecisionTreeClassifier
except:
    pass

try:
    from sklearn.metrics import precision_score
except:
    pass

try:
    from sklearn.metrics import recall_score
except:
    pass

try:
    from sklearn.metrics import f1_score
except:
    pass

try:
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
except:
    pass

try:
    from sklearn.neighbors import KNeighborsClassifier
except:
    pass

try:
    from sklearn.linear_model import LogisticRegression
except:
    pass

try:
    from sklearn.naive_bayes import GaussianNB
except:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
except:
    pass

try:
    from sklearn.svm import SVC
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

x_train, y_train, x_test, y_test = XtrainMLP, XtestMLP, ytrainMLP, ytestMLP


def keras_fmin_fnct(space):

    model = Sequential()
    model.add(Dense(100, input_shape=(n_cols,)))
    model.add(Dense(space['Dense'], activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam',loss='mean_squared_error')
    result = model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              epochs=2,
              verbose=2,
              validation_split=0.1)
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Dense': hp.choice('Dense', [256, 512, 1024]),
        'batch_size': hp.choice('batch_size', [64, 128]),
    }
