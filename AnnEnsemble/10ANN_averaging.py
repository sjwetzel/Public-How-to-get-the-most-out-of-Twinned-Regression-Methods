# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:12:43 2019
@author: anonymous
"""
#%tensorflow_version 1.2
from __future__ import print_function
    
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import tensorflow.keras
import itertools
from data.data import *
import sys
from progressbar import ProgressBar as PB
from SNN_helper import *
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Subtract,Input,Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from sklearn.linear_model import LinearRegression
from tensorflow.keras import backend as K
import argparse
parser = argparse.ArgumentParser(description='TNN runner')
parser.add_argument('--dataset', help='Options are: bostonHousing, concreteData, energyEfficiency, proteinStructure, randomFunction', dest='dataset', default='bostonHousing')
parser.add_argument('--dataset_path', help='Relative path to datasets', dest='dataset_path', default='./data/')
parser.add_argument('--val_pct', help='Percentage of validation split.', dest='val_pct', default=0.05, type=float)
parser.add_argument('--test_pct', help='Percentage of test split.', dest='test_pct', default=0.05, type=float)
parser.add_argument('--l2', help='L2 Regularization weighting.', dest='l2', default=0.0, type=float)
parser.add_argument('--seed', help='Random seed', dest='seed', default=13, type=int)
parser.add_argument('--num', help='Number of datapoints for random function', default=1000, type=int)
parser = parser.parse_args()
l2=parser.l2
Loop_weight=0.0
print(Loop_weight)
import time
print(parser.dataset)
start = time.time()

rmse_train_list=[]
rmse_val_list=[]
rmse_test_list=[]
rmse_test2_list=[]



### scaling factor 100,300,1000,3000,10000,30000,100000
n=1000
#(x_full, y_full) = getData(parser.dataset, parser.dataset_path)
#(x_full, y_full) = getData('randomFunction', './data/',n)
(x_full, y_full) = getData(parser.dataset, './data/',n)
#(x_full, y_full) = getData('testFunction', './data/',n)
NEIGHBORS_train=100
NEIGHBORS_inference=100
RANDOM_NEIGHBORS=False
TRAIN_ratio=0.7
VAL_ratio=0.1
TEST_ratio=1-TRAIN_ratio-VAL_ratio

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
for iteration in range(25):
    parser.seed=iteration
    # set the seed
    print('Random seed:', parser.seed)
    
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(parser.seed)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(parser.seed)
    
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #import tensorflow as tf
    tf.random.set_random_seed(parser.seed)
    
  

    #(x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=parser.val_pct, test_pct=parser.test_pct)
    (x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=VAL_ratio, test_pct=TEST_ratio,rand=True)
    print(x_train_single[0])
    
    ##### divide test set
    
    n_test_split=int(len(x_test_single)*1)
    x_test2_single=x_test_single
    y_test2_single=y_test_single   
    x_test_single=x_test_single
    y_test_single=y_test_single
    
    
    ##### center and normalize
    cn_transformer=CenterAndNorm()    
    
    
    x_train_single,y_train_single=cn_transformer.fittransform(x_train_single,y_train_single)
    x_val_single,y_val_single=cn_transformer.transform(x_val_single,y_val_single)
    x_test_single,y_test_single=cn_transformer.transform(x_test_single,y_test_single)
    x_test2_single,y_test2_single=cn_transformer.transform(x_test2_single,y_test2_single)

    ##########################################
    
    
    ############ NN parameters
    
    batch_size = 16
    epochs = 10000
    
    ############### create NN
    
    def getANet(seed):
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=(x_train_single.shape[1],),kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2)))
        model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2)))
        model.add(layers.Dense(1,kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2)))
    
        return model
    
    nets = [getANet(i) for i in range(1)]
    # model.summary()

    preds_train = []
    preds_val = []
    preds_test = []
    preds_test2 = []
    
    for model in nets:
    # Let's train the model 
      model.compile(optimizer=tensorflow.keras.optimizers.Adadelta(lr=1.0),
                  loss='mse',
                  metrics=['mse'])
    
      reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                    patience=20, verbose=1,min_lr=0)
    
    
      early_stop= EarlyStopping(monitor='val_loss', patience=50, verbose=1)
      mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    
      history = model.fit(x_train_single, y_train_single,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val_single, y_val_single),
                shuffle=True,
                callbacks=[reduce_lr,early_stop, mcp_save],
                verbose=0)
      #model.load_weights('mdl_wts.hdf5')
    
    

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
    
    
print("FINAL ann mean train rmse",np.mean(rmse_train_list),"+-",np.std(rmse_train_list))    
print("FINAL ann mean val rmse",np.mean(rmse_val_list),"+-",np.std(rmse_val_list))    
print("FINAL ann mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
print("FINAL ann mean test2 rmse",np.mean(rmse_test2_list),"+-",np.std(rmse_test2_list))

end = time.time()
print("time in seconds",end - start)