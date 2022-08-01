"""
   Copyright (C) 2022 ETH Zurich. All rights reserved.

   Author: Sergei Vostrikov, ETH Zurich

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

import sys
import os

# Insert path to library to system path
path_to_lib = '../../../'
print(path_to_lib)
sys.path.insert(0, path_to_lib)

from scripts.extract_features.extract_features_from_raw import extract_features_from_raw_data

if __name__ == "__main__":

    rf_dataset_path =  '../../data/rf_dataset.hdf5'

    extract_features_from_raw_data(rf_dataset_path, 
                                   path_to_save_features='../../../results')