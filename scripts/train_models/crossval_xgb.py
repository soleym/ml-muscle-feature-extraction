import numpy as np
import matplotlib.pyplot as plt

from time import asctime

import xgboost as xgb

import sys
import csv

from scipy import signal


# Insert path to repository, to import utility functions
path_to_utils ='../../'
sys.path.insert(0, path_to_utils)
from utils.pca_utils import print_crossval_scores, compute_scores

def cross_val_xgb(feats, penns, cv=4, n_estimators = 100, total_importance = 0.95):
    '''
    Cross validate the pca method for given data and split cv (default is 4)
    '''
    b, a = signal.butter(8, 0.04)
    all_rmse, all_max_error, all_mse, all_mae, all_r2 = [],[],[],[],[]
    
    step_size = len(penns)//cv
    for i in range(cv):
        print(str(i) + '    ' +str(asctime()))
        feat_val = feats[i*step_size:(i+1)*step_size]
        penn_val = penns[i*step_size:(i+1)*step_size]
        feat_train = np.delete(feats, np.arange(i*step_size,(i+1)*step_size), axis=0)
        penn_train = np.delete(penns, np.arange(i*step_size,(i+1)*step_size))
        
        if i == 0 or i == cv-1:
            penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
        else:
            # Smooth train before val
            penn_train[np.arange(0,i*step_size)] = signal.filtfilt(b,a, penn_train[np.arange(0,i*step_size)], padlen=150)
            # Smooth train after val
            penn_train[np.arange(i*step_size,len(penn_train))] = signal.filtfilt(b,a, penn_train[np.arange(i*step_size,len(penn_train))], padlen=150)

        penn_val = signal.filtfilt(b,a, penn_val, padlen=150)
        
        # Perform the method with current training and validation
        # Build a forest and compute the impurity-based feature importances
        forest = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 0)

        forest.fit(feat_train,penn_train)

        importances = forest.feature_importances_
        
        sum_importances = 0
        num_feats = 0
        while sum_importances < total_importance:
            num_feats += 1
            sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

        important_indices = np.argsort(importances)[-1*num_feats:]
        # Remove features of little importance
        feat_train_trimmed = feat_train[:,important_indices]
        feat_val_trimmed = feat_val[:,important_indices]
        
        reg = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 0)
        reg.fit(feat_train_trimmed, penn_train, verbose = False)
        
        # Predict pennation angles and get scores
        penn_predict = reg.predict(feat_val_trimmed)
        
        plt.figure(figsize = (8,8))
        #plt.plot(np.arange(len(penns)), penns, label = 'True data')
        #plt.plot(np.arange(i*step_size,(i+1)*step_size),penn_predict, label = 'Predicted pennation angles')

        plt.scatter(np.delete(np.arange(len(penns)), np.arange(i*step_size,(i+1)*step_size), axis=0), 
                 penn_train, label = 'Train', s = 2)
        
        plt.plot(np.arange(i*step_size,(i+1)*step_size), penn_val, label = 'Test')
        plt.plot(np.arange(i*step_size,(i+1)*step_size),penn_predict, label = 'Predicted pennation angles')
        
        plt.xlabel('Frame')
        plt.ylabel('Pennation_angle')
        plt.title('cv'+ str(i))
        plt.legend()
        plt.savefig('cv_'+ str(i)+'.png')
        
        np.savetxt('../../results/pennation_angle' +str(i)+'.csv', penn_predict, delimiter=',')
        rmse, max_err, mse, mae, r2 = compute_scores(penn_val, penn_predict) 

        all_rmse.append(rmse)
        all_max_error.append(max_err)
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
    
    return all_rmse, all_max_error, all_mse, all_mae, all_r2


print(asctime())

# Get the hand-crafted features
with open('../../results/ds_features_prepped_w8.csv', newline='') as csvfile: 
    feats = list(csv.reader(csvfile))

print(asctime())

feats = np.array(feats)
print(feats.shape)

# Get the ground truth pennation angles
pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',')

penns = pennation_angles
print(penns.shape)

print('Crossvalidate...')
print(asctime())
print()
rmse, max_err, mse, mae, r2 = cross_val_xgb(feats, penns, 
                                            cv=5, 
                                            n_estimators = 100, 
                                            total_importance = 0.95)
print(asctime())
print()


print_crossval_scores(rmse, mse, max_err, mae, r2)










