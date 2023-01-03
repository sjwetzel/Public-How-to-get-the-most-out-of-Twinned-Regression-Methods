# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:12:43 2019
@author: anonymous
"""
#%tensorflow_version 1.2
from __future__ import print_function
    

import matplotlib.pyplot as plt

import itertools
from data.data import *
import sys

from SNN_helper import *

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

import time

start = time.time()

rmse_train_list=[]
rmse_val_list=[]
rmse_test_list=[]
rmse_test2_list=[]  ### for knn test 2 = test 1

rmse_optimal_test=[]
optimal_nn=[]


### data sets
#'Biocon'
#'bostonHousing'
#'concreteData'
#'energyEfficiency'
#'RCL'
#'testFunction'
#'wine'
#'WheatStoneBridge'
#'yachtHydrodynamics'

### some datasets have a variable number of data points
n=1000

(x_full, y_full) = getData('bostonHousing', './data/',n)

NEIGHBORS_train=32
NEIGHBORS_inference=NEIGHBORS_train
RANDOM_NEIGHBORS=False
TRAIN_ratio=0.7
VAL_ratio=0.1
TEST_ratio=1-TRAIN_ratio-VAL_ratio

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
for iteration in range(25):     # loop over different train val test splits
    seed=iteration
   
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)
    

    
  

    (x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=VAL_ratio, test_pct=TEST_ratio,rand=True)
    print(x_train_single[0])
    
    ##### divide test set
    
    n_test_split=int(len(x_test_single)*1)
    x_test2_single=x_test_single
    y_test2_single=y_test_single   
    x_test_single=x_test_single
    y_test_single=y_test_single
    x_train_single=np.concatenate((x_train_single,x_val_single),axis=0    )    ### for knn we combine train and val bc of crossvalidation
    y_train_single=np.concatenate((y_train_single,y_val_single),axis=0    )
    
    ##### center and normalize
    cn_transformer=CenterAndNorm()    
    
    
    x_train_single,y_train_single=cn_transformer.fittransform(x_train_single,y_train_single)
    x_val_single,y_val_single=cn_transformer.transform(x_val_single,y_val_single)
    x_test_single,y_test_single=cn_transformer.transform(x_test_single,y_test_single)
    x_test2_single,y_test2_single=cn_transformer.transform(x_test2_single,y_test2_single)

    ##########################################
    


    # model.summary()

    preds_train = []
    preds_val = []
    preds_test = []
    preds_test2 = []
    
    for i in range(1):  # create an ensemble of predictors
    # Let's train the model 
      model=KNeighborsRegressor(n_neighbors=NEIGHBORS_train)
      model.fit(x_train_single,y_train_single)
    
    

      preds_train.append(model.predict(x_train_single))
      preds_val.append(model.predict(x_val_single))
      preds_test.append(model.predict(x_test_single))
      preds_test2.append(model.predict(x_test2_single))


    
    preds_train = np.array(preds_train)
    preds_train_av = np.mean(preds_train, axis=0)
    preds_train_av = preds_train_av.reshape(preds_train_av.shape[0])
    
    preds_val = np.array(preds_val)
    preds_val_av = np.mean(preds_val, axis=0)
    preds_val_av = preds_val_av.reshape(preds_val_av.shape[0])

    preds_test = np.array(preds_test)
    preds_test_av = np.mean(preds_test, axis=0)
    preds_test_av = preds_test_av.reshape(preds_test_av.shape[0])

    preds_test2 = np.array(preds_test2)
    preds_test2_av = np.mean(preds_test2, axis=0)
    preds_test2_av = preds_test2_av.reshape(preds_test2_av.shape[0])
    
    Y_mse_train=  (preds_train_av - y_train_single)**2   *(cn_transformer.Ymax)**2
    Y_mse_val=  (preds_val_av - y_val_single)**2   *(cn_transformer.Ymax)**2
    Y_mse_test=  (preds_test_av - y_test_single)**2   *(cn_transformer.Ymax)**2
    Y_mse_test2=  (preds_test2_av - y_test2_single)**2   *(cn_transformer.Ymax)**2
    
    print('Train RMSE:', np.average(Y_mse_train)**0.5)
    print('Val RMSE:',np.average(Y_mse_val)**0.5)
    print('Test RMSE:',np.average(Y_mse_test)**0.5)
    print('Test2 RMSE:',np.average(Y_mse_test2)**0.5)    
    
    
    trainrmse=np.average(Y_mse_train)**0.5
    valrmse=np.average(Y_mse_val)**0.5
    testrmse=np.average(Y_mse_test)**0.5
    test2rmse=np.average(Y_mse_test2)**0.5
  
    rmse_train_list.append(trainrmse)
    rmse_val_list.append(valrmse)    
    rmse_test_list.append(testrmse)
    rmse_test2_list.append(test2rmse)
    
    print("prelim mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
    
    
    knn2 = KNeighborsRegressor()

    param_grid = {'n_neighbors': np.arange(1, 32)}
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    knn_gscv.fit(x_train_single, y_train_single)
    
    print('best neighbors:',knn_gscv.best_params_)
    rmse=np.sqrt(np.average((knn_gscv.predict(x_test_single)-y_test_single)**2))
    print('optimal test rmse:',rmse)
    
    rmse_optimal_test.append(rmse)
    optimal_nn.append(knn_gscv.best_params_['n_neighbors'])
    
    
print("FINAL knn mean train rmse",np.mean(rmse_train_list),"+-",np.std(rmse_train_list))    
print("FINAL knn mean val rmse",np.mean(rmse_val_list),"+-",np.std(rmse_val_list))    
print("FINAL knn mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
#print("FINAL knn mean test2 rmse",np.mean(rmse_test2_list),"+-",np.std(rmse_test2_list))


print("optimal knn mean test rmse",np.mean(rmse_optimal_test),"+-",np.std(rmse_optimal_test))
print("optimal neighbors",np.mean(optimal_nn),"+-",np.std(optimal_nn))

end = time.time()
print("time in seconds",end - start)

