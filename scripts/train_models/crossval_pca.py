import numpy as np

from time import asctime

import csv
import sys


# Insert path to repository, to import utility functions
path_to_utils ='../../'
sys.path.insert(0, path_to_utils)
from utils.pca_utils import cross_val_pca, print_crossval_scores

if __name__ == '__main__':

    # Get the data
    print('Get data...')
    print(asctime())
    print()
    
    # Get the hand-crafted features
    with open('../../results/ds_features_prepped_w8.csv', newline='') as csvfile:
        feats = list(csv.reader(csvfile))
    feats = np.array(feats)
    print(feats.shape)
    
    # Get the ground truth pennation angles
    pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',')
    penns = np.array(pennation_angles)
    print(penns.shape)
    
    print('Crossvalidate...')
    print(asctime())
    print()
    rmse, max_err, mse, mae, r2 = cross_val_pca(feats, penns, 
                                                cv=5, 
                                                n_estimators = 100, 
                                                total_importance = 0.95, 
                                                plot_figures = True, 
                                                verbose = 1, save_path = '../../results/')
    print(asctime())
    print()
    
    print_crossval_scores(rmse, mse, max_err, mae, r2)
