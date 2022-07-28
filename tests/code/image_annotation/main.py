import sys
import os

# Insert path to library to system path
path_to_lib = '../../../'
print(path_to_lib)
sys.path.insert(0, path_to_lib)

from scripts.extract_features.extract_features_from_images import extract_features_from_images

if __name__ == "__main__":

    image_dataset_path =  '../../data/image_dataset.hdf5'

    extract_features_from_images(image_dataset_path, 
                                 path_to_save_results='../../../results', 
                                 save_images=True)