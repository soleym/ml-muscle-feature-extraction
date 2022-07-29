import numpy as np
import pickle 

from time import asctime

import xgboost as xgb

import csv

from scipy import signal


def train_handcrafted_xgb(feats, penns, n_estimators = 100, total_importance = 0.95, model_save_path = None):
    '''
    Train the handcrafted xgb method 
    '''
    print('Training the model...')
    print(asctime())
    print()
    
    b, a = signal.butter(8, 0.04)

    
    penns = signal.filtfilt(b,a, penns, padlen=150)
    
    # Build a forest and compute the impurity-based feature importances
    forest = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 0)

    forest.fit(feats,penns)

    importances = forest.feature_importances_
    
    sum_importances = 0
    num_feats = 0
    while sum_importances < total_importance:
        num_feats += 1
        sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

    important_indices = np.argsort(importances)[-1*num_feats:]
    # Remove features of little importance
    feat_trimmed = feats[:,important_indices]
    
    reg = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 0)
    reg.fit(feat_trimmed, penns, verbose = False)
    
    if model_save_path:
        print(asctime())
        print('Saving the model to: ', model_save_path)
        
        forest_save = model_save_path + 'handcrafted_forest.dat'
        pickle.dump(forest, open(forest_save, 'wb'))
        
        print(asctime())


if __name__ == '__main__':
    
    print(asctime())
    
    # Get the hand-crafted features
    with open('../../results/ds_features_prepped_w8.csv', newline='') as csvfile: 
        feats = list(csv.reader(csvfile))
    
    print(asctime())
    
    feats = np.array(feats)
    
    # Get the ground truth pennation angles
    pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',')
    
    penns = pennation_angles
    
    train_handcrafted_xgb(feats, penns, n_estimators = 100, total_importance = 0.95,model_save_path='../../results/')

    









