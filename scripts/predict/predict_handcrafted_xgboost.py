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
import torch
import joblib
import csv

from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

def predict_handcrafted_xgboost(feats, load_path = '../../results/', save_path = None):
    
    feats = np.array(feats)
    
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)  
    
    print('Loading model...')
    print(asctime())
    print()
    
    forest = joblib.load(load_path + 'handcrafted_forest.joblib.dat')
    reg    = joblib.load(load_path + 'handcrafted_regressor.joblib.dat')

    print('Making predictions...')
    print(asctime())
    print()
    
    importances = forest.feature_importances_
        
    sum_importances = 0
    num_feats = 0
    while sum_importances < 0.95:
        num_feats += 1
        sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

    important_indices = np.argsort(importances)[-1*num_feats:]
    # Remove features of little importance
    feat_trimmed = feats[:,important_indices]
    
    # Predict pennation angles and get scores
    penn_predict = reg.predict(feat_trimmed)

    if save_path:
        np.savetxt(save_path + 'predicted_handcrafted_xgb_pennation_angle.csv', np.array(penn_predict), delimiter=',')
   
    return penn_predict
    
    
if __name__ == "__main__":
    
    # Get the hand-crafted features.
    # We assume that they have already been extracted for this data, using /scripts/extract_features/extract_features_from_raw.py
    # and the link should point to the correct file 
    with open('../../results/ds_features_prepped_w8.csv', newline='') as csvfile: 
        feats = list(csv.reader(csvfile))
    
    predict_handcrafted_xgboost(feats, load_path = '../../results/', save_path = None)
    
    
