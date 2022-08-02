"""
   Copyright (C) 2022 ETH Zurich. All rights reserved.

   Author: Soley Hafthorsdottir, ETH Zurich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
       
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

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
