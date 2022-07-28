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