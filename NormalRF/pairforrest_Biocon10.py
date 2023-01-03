import argparse
import numpy as np
import pandas as pd
from data import *
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import time



# Load Data ------------------------------------------
parser = argparse.ArgumentParser(description='RF Runner')
parser.add_argument('--dataset', help='Options are: bostonHousing, concreteData, \
                    energyEfficiency, yachtHydrodynamics', 
                    dest='dataset', default='yachtHydrodynamics')

parser.add_argument('--Nrand',  dest='Nrand', type=int, default=1)                  
parser = parser.parse_args()

X,y=getData(key=parser.dataset, path = './data/', n_points=1000)
X,y = X[:200], y[:200]

#Calculate the root mean square error on predictions
def score(y_predictions, y_test):
  score=np.sqrt(np.sum((y_predictions - y_test)**2)/len(y_test)) 
  return score



#Feature Expansion
def feature_expansion( data, target, k_rand=None, random_state=None ):
  n=len(data)
  m=len(data.T)
  if k_rand==None:  # Full Expansion if k-rand = None
    FE_X = np.zeros((n**2,3*m))
    FE_y = np.zeros((n**2,1))
    for i in range(n):  
      for j in range(n):
        FE_X[i*n+j]=np.concatenate((data[i],data[j],data[i]-data[j]))
        FE_y[i*n+j]=target[i]-target[j]
    return FE_X, FE_y

  else: # Use k_rand Anchors
    FE_X=np.zeros((n*k_rand,3*m))
    FE_y=np.zeros((n*k_rand,1))
    for i in range(n):
      np.random.seed(i+random_state) # It should be in the for loop otherwise random.randint will pick always the same elements to pair with
      idx = np.random.choice(a=np.arange(n), size=k_rand, replace=False) #Pick k_rand Random Points For Pairing to Each Element 
      for j in range(k_rand):
        FE_X[i*k_rand+j]=np.concatenate((data[i],data[idx[j]],data[i]-data[idx[j]]))
        FE_y[i*k_rand+j]=target[i]-target[idx[j]]

    return FE_X, FE_y

'''
Predict the Label
tree - Decision Tree which is trained on expanded data - feature_expansion(data,target)
X_new - New Data on which we have to predict the labels
data - X_train_lab basically (Note: Unexpanded) 
target -  y_train_lab basically (Note: Unexpanded)
'''
def expanded_predict(model, X_new, data, target):
  n_new, n_old = len(X_new), len(data)
  y_labels=np.zeros(n_new)
  for i in range(n_new):
    stacked_Xi = np.tile(X_new[i],(n_old,1))
    X_to_predict = np.concatenate((stacked_Xi, data, stacked_Xi - data),axis=1)
    y_preds = target + model.predict(X_to_predict)
    y_labels[i]=np.mean(y_preds)
  return y_labels





param_grid = { 
    'max_depth': [ 8, 16, 32],  ### 16, 32, rarely 8 [ 4, 8, 16, 32, 64]
    'max_features': [0.33, 0.667, 1.0],  ### all
    'min_samples_leaf': [1,2],    ### 1,2 never 5 [1, 2, 5]
    'min_samples_split': [2,3, 4],    ### 2,4  very rarely 8 [2, 4, 8]
    'n_estimators': [100]}    ### set to 100? [100, 300, 600]


pre_parameters = []
parameters = []
pre_scores = []
tw_scores = []
N_rand = parser.Nrand

for i in range(1): 
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.16, shuffle=True, random_state = N_rand - 1) # Train - Rest 80 - 20
  #X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, shuffle=True, random_state = N_rand - 1) # Val - Test 10 - 10
  X_lab, X_nlab, y_train_lab, y_train_nlab = train_test_split(X, y, shuffle=True, test_size=0.5, random_state = N_rand - 1) # Labelled - Non-Labelled 40 - 40

  pre_rfr=RFR(random_state = N_rand - 1) #Create DecisionTree
  pre_clf = GridSearchCV(pre_rfr, param_grid, cv= 5)
  pre_clf.fit( X_lab, y_train_lab)
  pre_best_params = pre_clf.best_params_  
  pre_parameters.append(pre_best_params)
  pre_best_rfr =  RFR(n_estimators =  pre_best_params['n_estimators'],
                max_features = pre_best_params['max_features'],
                max_depth = pre_best_params['max_depth'],
                min_samples_leaf = pre_best_params['min_samples_leaf'],
                min_samples_split = pre_best_params['min_samples_split'],
                random_state = N_rand - 1,
                n_jobs = -1)
  pre_best_rfr.fit( X_lab, y_train_lab)
  pre_y_preds = pre_best_rfr.predict(X_nlab)
  pre_sc = score(pre_y_preds, y_train_nlab)
  pre_scores.append(pre_sc)


  FE_X, FE_y = feature_expansion(X_lab, y_train_lab,  random_state = i) #Do the feature expansion on labelled set
  rfr=RFR(random_state = N_rand - 1) #Create DecisionTree
  clf = GridSearchCV(rfr, param_grid, cv= 5)
  clf.fit(FE_X, FE_y.reshape(-1))
  best_params = clf.best_params_  
  parameters.append(best_params)
  best_rfr =  RFR(n_estimators =  best_params['n_estimators'],
                max_features = best_params['max_features'],
                max_depth = best_params['max_depth'],
                min_samples_leaf = best_params['min_samples_leaf'],
                min_samples_split = best_params['min_samples_split'],
                random_state = N_rand - 1,
                n_jobs = -1)
  best_rfr.fit(FE_X, FE_y.reshape(-1))
  y_preds = expanded_predict(best_rfr, X_nlab, X_lab, y_train_lab )
  sc = score(y_preds, y_train_nlab)
  tw_scores.append(sc)

print("--------------------------------------------------------------------------------------------", flush = True)
print(f"Epoch {N_rand - 1 + 1}", flush=True)
print(pre_parameters, flush=True)
print(parameters, flush=True)
print(f"Norm {np.array(pre_scores).mean()}", flush=True)
print(f"Twinned {np.array(tw_scores).mean()}", flush=True)
print("--------------------------------------------------------------------------------------------", flush=True)
