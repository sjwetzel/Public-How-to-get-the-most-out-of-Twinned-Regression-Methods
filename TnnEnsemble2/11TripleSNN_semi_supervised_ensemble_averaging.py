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

Loop_weight=0.0
print(Loop_weight)
import time
print(parser.dataset)
start = time.time()

rmse_train_list=[]
rmse_val_list=[]
rmse_test_list=[]
rmse_test2_list=[]



TRAIN_ratio=0.7
VAL_ratio=0.1
TEST_ratio=1-TRAIN_ratio-VAL_ratio

### scaling factor 100,300,1000,3000,10000,30000,100000
n=1000

#(x_full, y_full) = getData(parser.dataset, parser.dataset_path)
#(x_full, y_full) = getData('wine', './data/',n)
(x_full, y_full) = getData(parser.dataset, './data/',n)
#(x_full, y_full) = getData('testFunction', './data/',n)

####### this is just for preloading to make sure Neighbors are calculated properly
(x_train_single, y_train_single), (x_val_single, y_val_single), (x_test_single, y_test_single) = splitData(x_full, y_full, val_pct=VAL_ratio, test_pct=TEST_ratio,rand=True)

#######NEIGHBORS_train=len(x_train_single)-2


NEIGHBORS_train=len(x_train_single)-2
NEIGHBORS_inference=len(x_train_single)-2


RANDOM_NEIGHBORS=True



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
    
    n_test_split=int(len(x_test_single)*5/6)
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
    epochs = 2000
    
    preds_train = []
    preds_val = []
    preds_test = []
    preds_test2 = []
    
    for net in range(1):
    
        ############### create NN
        
        observer_a=Input(shape=(x_train_single.shape[-1],),name='observer_a')
        observer_b=Input(shape=(x_train_single.shape[-1],),name='observer_b')
        
        l2=parser.l2
        
        merged_layer = tensorflow.keras.layers.Concatenate()([observer_a, observer_b])
        
        merged_layer=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
        merged_layer=Dense(128,activation='relu',name='iout',kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
        output=Dense(1,kernel_regularizer=regularizers.l2(l2),activity_regularizer=regularizers.l2(l2))(merged_layer)
        model = Model(inputs=[observer_a, observer_b], outputs=output)
        #model.summary()
        
        
        
        observer_c=Input(shape=(x_train_single.shape[-1],),name='observer_c')
        observer_d=Input(shape=(x_train_single.shape[-1],),name='observer_d')
        observer_e=Input(shape=(x_train_single.shape[-1],),name='observer_e')
        
        
        
        SNN1=model([observer_c,observer_d])
        SNN2=model([observer_d,observer_e])
        SNN3=model([observer_e,observer_c])
        
        tri_layer = tensorflow.keras.layers.Concatenate()([SNN1,SNN2,SNN3])
        
        
        TRImodel= Model(inputs=[observer_c, observer_d, observer_e], outputs=tri_layer)
        #TRImodel.summary()
        
        #### custom loss
        def custom_loss(y_true,y_pred):
            #eps=1.0e-6
            #y_pred=K.clip(y_pred,eps,1.0-eps)
            
            half=int(batch_size//2)
           
            y_case=tf.constant(([1.]*half)+([0.]*half))
            y_case_odd=tf.constant(([0.]*half)+([1.]*half))
            
            loss_A=K.mean(K.square(y_pred - y_true), axis=-1)
            loss_B=K.square(K.mean(y_pred,axis=-1))      
    
            return K.in_train_phase(y_case*loss_A[:batch_size]+Loop_weight*y_case_odd*loss_B[:batch_size], loss_A)
        
        # Let's train the model 
        TRImodel.compile(loss=custom_loss
                      ,optimizer=tensorflow.keras.optimizers.Adadelta(lr=1)
                     ,metrics=['mse']
                      )
        
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=int(1000/np.sqrt(n)+1), verbose=1,min_lr=0)
        
        
        early_stop= EarlyStopping(monitor='val_loss', patience=2*int(1000/np.sqrt(n)+1), verbose=0)
        mcp_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=0)
        history = TRImodel.fit_generator(nearest_neighbor_triple_semi_supervised_generator_all_excluding_labelled_loops(x_train_single, y_train_single, np.concatenate((x_test_single,x_val_single),axis=0), batch_size,NEIGHBORS_train,RANDOM_NEIGHBORS),
                                        steps_per_epoch=(len(x_train_single)+len(x_test_single))*5/batch_size,
                                        epochs=epochs,
                                        validation_data=nearest_neighbor_triple_supervised_generator_loops_VAL(x_val_single, y_val_single, len(x_val_single), int(NEIGHBORS_train*VAL_ratio/TRAIN_ratio), RANDOM_NEIGHBORS), ## scale the number of neighbors to match the average distance in the training set
                                        validation_steps=1,##len(x_val_single)*100/batch_size,
                                        callbacks=[reduce_lr,early_stop, mcp_save],verbose=0)
        
        #TRImodel.load_weights('mdl_wts.hdf5') #this will lead to overfitting to the validation data
        
        
        # Plot training & test loss values
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Val'], loc='upper left')
        # # plt.show()
        # plt.savefig('loss.pdf')
            
    
        
        #####################
    
        Y_pred_train=[]
        Y_pred_r_train=[]
        Y_pred_check_train=[]
        Y_median_train=[]
        Y_var_train=[]
        Y_mse_train=[]
        
        
        neighborIndices=nearestNeighbors(x_train_single,x_train_single,NEIGHBORS_inference,RANDOM_NEIGHBORS)
        for i in range(len(x_train_single)):
            
            ## new
            pair_B=np.array([x_train_single[i]]*NEIGHBORS_inference)        
                    
            diffA=model.predict([pair_B,x_train_single[neighborIndices[i]]]).flatten()
            diffB=model.predict([x_train_single[neighborIndices[i]],pair_B]).flatten()
            ##
    
            Y_pred_train.append(np.average(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]], weights=None))

            Y_pred_check_train.append(np.var(0.5*diffA+0.5*diffB))
    
            Y_median_train.append(np.median(diffA+y_train_single[neighborIndices[i]]))
            Y_var_train.append(np.var(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]]))
        
        
        #####################
    
        Y_pred_val=[]
        Y_pred_r_val=[]
        Y_pred_check_val=[]
        Y_median_val=[]
        Y_var_val=[]
        Y_mse_val=[]
        
        
        neighborIndices=nearestNeighbors(x_val_single,x_train_single,NEIGHBORS_inference,RANDOM_NEIGHBORS)
        for i in range(len(x_val_single)):
            ## new
            pair_B=np.array([x_val_single[i]]*NEIGHBORS_inference)        
                    
            diffA=model.predict([pair_B,x_train_single[neighborIndices[i]]]).flatten()
            diffB=model.predict([x_train_single[neighborIndices[i]],pair_B]).flatten()
            ##
    
            Y_pred_val.append(np.average(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]], weights=None))

            Y_pred_check_val.append(np.var(0.5*diffA+0.5*diffB))
    
            Y_median_val.append(np.median(diffA+y_train_single[neighborIndices[i]]))
            Y_var_val.append(np.var(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]]))

        
        
        #####################
        
        Y_pred_test=[]
        Y_pred_r_test=[]
        Y_pred_check_test=[]
        Y_median_test=[]
        Y_var_test=[]
        Y_mse_test=[]
        
        
        neighborIndices=nearestNeighbors(x_test_single,x_train_single,NEIGHBORS_inference,RANDOM_NEIGHBORS)
        for i in range(len(x_test_single)):
            ## new
            pair_B=np.array([x_test_single[i]]*NEIGHBORS_inference)      
                    
            diffA=model.predict([pair_B,x_train_single[neighborIndices[i]]]).flatten()
            diffB=model.predict([x_train_single[neighborIndices[i]],pair_B]).flatten()
            ##
    
            Y_pred_test.append(np.average(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]], weights=None))

            Y_pred_check_test.append(np.var(0.5*diffA+0.5*diffB))
    
            Y_median_test.append(np.median(diffA+y_train_single[neighborIndices[i]]))
            Y_var_test.append(np.var(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]]))

        
        
        #####################
        
        Y_pred_test2=[]
        Y_pred_r_test2=[]
        Y_pred_check_test2=[]
        Y_median_test2=[]
        Y_var_test2=[]
        Y_mse_test2=[]
        
        
        neighborIndices=nearestNeighbors(x_test2_single,x_train_single,NEIGHBORS_inference,RANDOM_NEIGHBORS)
        for i in range(len(x_test2_single)):
            ## new
            pair_B=np.array([x_test2_single[i]]*NEIGHBORS_inference)       
                    
            diffA=model.predict([pair_B,x_train_single[neighborIndices[i]]]).flatten()
            diffB=model.predict([x_train_single[neighborIndices[i]],pair_B]).flatten()
            ##
    
            Y_pred_test2.append(np.average(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]], weights=None))

            Y_pred_check_test2.append(np.var(0.5*diffA+0.5*diffB))
    
            Y_median_test2.append(np.median(diffA+y_train_single[neighborIndices[i]]))
            Y_var_test2.append(np.var(0.5*diffA-0.5*diffB+y_train_single[neighborIndices[i]]))

        
        
        
        preds_train.append(Y_pred_train)
        preds_val.append(Y_pred_val)
        preds_test.append(Y_pred_test)
        preds_test2.append(Y_pred_test2)
        
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
    
    
print("FINAL tnn mean train rmse",np.mean(rmse_train_list),"+-",np.std(rmse_train_list))    
print("FINAL tnn mean val rmse",np.mean(rmse_val_list),"+-",np.std(rmse_val_list))    
print("FINAL tnn mean test rmse",np.mean(rmse_test_list),"+-",np.std(rmse_test_list))
print("FINAL tnn mean test2 rmse",np.mean(rmse_test2_list),"+-",np.std(rmse_test2_list))

end = time.time()
print("time in seconds",end - start)