import numpy as np
from skimage.transform import rescale, resize
import cv2 as cv

import time
from time import asctime

import sys
import os 

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader, ImageLoader
from pybf.pybf.visualization import log_compress

from utils.image_labeling import get_reference_max, plot_double_results
from scripts.extract_features.extract_features_from_images import extract_features_from_images
from scripts.predict.predict_ae_xgboost import predict_ae_xgboost

if __name__ == "__main__":
    
    # Create folders to store visualizations and results (optional, can be modified as desired)
    if not os.path.exists('../../../results/frames_AE'):
        os.makedirs('../../../results/frames_AE')
        
    if not os.path.exists('../../../results/testdataset_ae_results'):
        os.makedirs('../../../results/testdataset_ae_results')
        
    if not os.path.exists('../../../results/testdataset_ALT_results'):
        os.makedirs('../../../results/testdataset_ALT_results')
    
    dataset_path = '../../data/rf_dataset.hdf5'
    image_dataset_path = '../../data/image_dataset.hdf5'
    
    data_loader_obj = DataLoader(dataset_path)
    # Both n_frame and m_acq start from 0, not 1 (like for image_dataloader)
    num_frames = data_loader_obj._num_of_frames
    num_acquis = data_loader_obj._num_of_acq_per_frame 
    # Get and prepare data
    print('Getting raw data')
    print()
    data = []
    t1 = time.time()
    for frame in range(num_frames):
        for acq in range(num_acquis):
            dt = data_loader_obj.get_rf_data(n_frame = frame, m_acq = acq)
            data.append(dt.T)
    print('Time to extract all data in seconds: ' + str(time.time()-t1))    
    
    
    data = np.array(data)
    data = data[:,:800,::2] # Throw away unimportant samples
    
    penn_predict = predict_ae_xgboost(data, load_path = '../../../', save_path = '../../../results/testdataset_ae_results')
    
    print('Plotting results...')
    print(asctime())
    print()
    
    th1, th2, penn, deep, pos_fas, point_deep, point_fas = extract_features_from_images(image_dataset_path, path_to_save_results='../../../results/testdataset_ALT_results', save_images=False)
    
    # Get the images and plot results
    img_loader_obj = ImageLoader(image_dataset_path)
    # Iterate over images and process them
    for frame_index in np.arange(0,len(img_loader_obj.frame_indices)):
        if frame_index % 50 == 0 or frame_index == 1:
            print(time.asctime() + '    Frame ' + str(frame_index))

        img_data = img_loader_obj.get_high_res_image(frame_index)

        # Calculate image sizes
        pixels_coords = img_loader_obj.get_pixels_coords()
        image_size_x_0 = pixels_coords[0, :].min()
        image_size_x_1 = pixels_coords[0, :].max()
        image_size_z_0 = pixels_coords[1, :].min()
        image_size_z_1 = pixels_coords[1, :].max()

        attenuate = True
        att = ''
        reference_max = None
        if attenuate == True:
            att = 'with saturated maximums '
            reference_max = get_reference_max(img_data, 0.2)

        db_range = 50

        image_scaled = rescale(log_compress(resize(np.abs(img_data), 
                                            (np.around(np.abs(image_size_z_1-image_size_z_0),4)*1e4, 
                                            np.around(np.abs(image_size_x_1-image_size_x_0),4)*1e4)),
                                            db_range, reference_max = reference_max), scale=1.0, mode='reflect', multichannel=False)

        # Plot the results
        img_rgb = plot_double_results(image_scaled, th1[frame_index], th2[frame_index], 
                                      penn_predict[frame_index],
                                      image_scaled.shape[0]- pos_fas[frame_index], point_fas, 
                                      image_scaled.shape[0]- deep[frame_index], point_deep, 
                                      frame_index)
        
        cv.imwrite('../../../results/frames_AE/frame-%04d'%frame_index + '.png', img_rgb)
    
    
    print(asctime())
    print()
