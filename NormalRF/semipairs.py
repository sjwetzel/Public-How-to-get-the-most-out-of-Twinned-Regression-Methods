import argparse
import numpy as np
import pandas as pd
from data import *
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import time



parser = argparse.ArgumentParser(description='RF Runner')
parser.add_argument('--dataset', help='Options are: bostonHousing, concreteData, \
                    energyEfficiency, yachtHydrodynamics', 
                    dest='dataset', default='yachtHydrodynamics')
                  
parser.add_argument('--Nrand',  dest='Nrand', type=int, default=0)
parser = parser.parse_args()

X,y=getData(key=parser.dataset, path = './data/', n_points=1000)



#Calculate the root mean square error on predictions
def score(y_predictions, y_test):
  score=np.sqrt(np.sum((y_predictions - y_test)**2)/len(y_test)) 
  return score

#Evaluate the split
def eval(data, target, feature, value):
  mask=data[:,feature]>value
  yR=target[mask]
  yL=target[np.logical_not(mask)]
  return (len(yR)*np.std(yR)+len(yL)*np.std(yL))/len(target)

#Feature Expansion
def feature_expansion( data, target, k_rand=None, random_state=None ):
  n=len(data)
  m=len(data.T)
  if k_rand==None:
    FE_X = np.zeros((n**2,3*m))
    FE_y = np.zeros((n**2,1))
    for i in range(n):  
      for j in range(n):
        FE_X[i*n+j]=np.concatenate((data[i],data[j],data[i]-data[j]))
        FE_y[i*n+j]=target[i]-target[j]
    return FE_X, FE_y

  else:
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
    'max_depth': [ 4, 8, 16, 32, 64],
    'max_features': [0.33, 0.667, 1.0],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 4, 8],
    'n_estimators': [100, 300, 600]}


N_splits = 1
parameters = []


iter = 1
gamma_scores=[]
couplings = [30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0] 
N_rand = parser.Nrand
n_add = round(X.shape[0] * 0.4 / 3 ) #Number of additions 


for i in range(1): 
  print(f'Epoch - {i+1}')
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle=True, random_state = N_rand - 1) # Train - Rest 80 - 20
  #X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, shuffle=True, random_state = N_rand - 1) # Val - Test 10 - 10
  X_lab, X_nlab, y_train_lab, y_train_nlab = train_test_split(X_train, y_train, shuffle=True, test_size=0.5, random_state = N_rand - 1) # Labelled - Non-Labelled 40 - 40

  FE_X, FE_y = feature_expansion(X_lab, y_train_lab, k_rand = 10, random_state = N_rand - 1) #Do the feature expansion on labelled set

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
                random_state = N_rand-1,
                n_jobs = -1)

  scores = []
  for gamma in couplings: 
    #Select Triplets
    idx = np.arange( 0, len( X_nlab ))
    np.random.seed(N_rand - 1)
    triplets = np.random.choice( a = idx, size = ( n_add , 3), replace = True)
    #########################

    curr_FE_X = FE_X.copy()
    curr_FE_y = FE_y.copy()

    accuracy=[]

    for j in range(iter+1):
      best_rfr.fit(curr_FE_X,curr_FE_y.reshape(-1))
      y_test_preds = expanded_predict(best_rfr, X_test, X_lab, y_train_lab)
      accuracy.append( score( y_test_preds, y_test ) )
      
      to_modify_FE_X = FE_X.copy()
      to_modify_FE_y = FE_y.copy()
  
      #Label non-labelled and add to the labelled data
      for trio in triplets:
        x1, x2, x3 = X_nlab[trio[0]], X_nlab[trio[1]], X_nlab[trio[2]]
        x1_x2 = np.concatenate( (x1, x2, x1 - x2) ).reshape(1,-1)
        x2_x3 = np.concatenate( (x2, x3, x2 - x3) ).reshape(1,-1)
        x3_x1 = np.concatenate( (x3, x1, x3 - x1) ).reshape(1,-1)
        
        y1_y2 = best_rfr.predict( x1_x2 ).reshape(1,1)
        y2_y3 = best_rfr.predict( x2_x3 ).reshape(1,1)
        y3_y1 = best_rfr.predict( x3_x1 ).reshape(1,1)

        a = y1_y2 + y2_y3 + y3_y1

        to_modify_FE_X = np.concatenate( ( to_modify_FE_X, x1_x2, x2_x3, x3_x1 ), axis=0 ) 
        to_modify_FE_y = np.concatenate( ( to_modify_FE_y, y1_y2 - gamma*a, y2_y3 - gamma*a, y3_y1 - gamma*a ), axis=0 ) 

      curr_FE_X = to_modify_FE_X
      curr_FE_y = to_modify_FE_y
    
    
    scores.append(accuracy)

    
  gamma_scores.append(scores)



scoresheet=np.zeros((1,len(couplings),iter+1))
means=np.zeros((len(couplings),iter+1))
stds=np.zeros((len(couplings),iter+1))
for i in range(1):
  for j in range(len(couplings)):
    for k in range(iter+1):
      scoresheet[i][j][k]=gamma_scores[i][j][k]
      
for j in range(len(couplings)):
  for k in range(iter+1):
    means[j][k]=scoresheet[:,j,k].mean()
    stds[j][k]=scoresheet[:,j,k].std()

print("------------------------------------------------------------------------------")
print(parameters)

for i in range(len(couplings)):
  print(f'normal score = {means[i][0]}')  
  print(f'gamma = {couplings[i]}, score = {means[i][1]} ')

print("------------------------------------------------------------------------------")
