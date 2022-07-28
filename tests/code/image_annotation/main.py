if __name__ == "__main__":

    image_dataset_path =  '../../tests/data/image_dataset.hdf5'

    extract_features_from_images(image_dataset_path, 
                                 path_to_save_results='../../results', 
                                 save_images=True)