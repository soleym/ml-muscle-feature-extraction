import numpy as np
import cv2 as cv

from skimage.transform import rescale, resize

import time
import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import ImageLoader
from pybf.pybf.visualization import log_compress

# Insert path to repository, to import utility functions
path_to_utils ='../../'
sys.path.insert(1, path_to_utils)
from utils.image_labeling import get_reference_max, plot_double_results

# Useful code to suppress output, from:  ######################################
# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
from contextlib import contextmanager
import os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
###############################################################################
# More useful code to suppress warnings from miniconda, from
# https://stackoverflow.com/questions/61444572/ignore-all-warnings-from-a-module          
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
###############################################################################

dataset_number = 1

if dataset_number == 1:
    # Stored on 'elendil' machine
    #image_dataset_path = '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session2/20191218_run_2_invivo_dynamic/1_1_full_aperture_pw_0/image_dataset.hdf5'
    # Stored on 'ojos3' machine
    image_dataset_path =  '/ojos3/scratch/session2_run2_pw0/image_dataset.hdf5' # Dataset 1    

    pennation_angles = np.loadtxt('../../results/sess2_run2_pw0/pennation_angle.csv', delimiter=',')
    theta_1 = np.loadtxt('../../results/sess2_run2_pw0/theta_1.csv', delimiter=',')
    theta_2 = np.loadtxt('../../results/sess2_run2_pw0/theta_2.csv', delimiter=',')
    pos_fas = np.loadtxt('../../results/sess2_run2_pw0/pos_fas.csv', delimiter=',')
    
if dataset_number == 2:
    # Stored on 'elendil' machine
    image_dataset_path = '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_3_invivo_dynamic/1_1_full_aperture_pw_0/image_dataset.hdf5'
    # Stored on 'ojos3' machine
    image_dataset_path =  '/ojos3/scratch/session1_run3_pw0/image_dataset.hdf5' # Dataset 2    

    pennation_angles = np.loadtxt('../../results/sess2_run2_pw0/pennation_angle.csv', delimiter=',')
    theta_1 = np.loadtxt('../../results/sess2_run2_pw0/theta_1.csv', delimiter=',')
    theta_2 = np.loadtxt('../../results/sess2_run2_pw0/theta_2.csv', delimiter=',')
    pos_fas = np.loadtxt('../../results/sess2_run2_pw0/pos_fas.csv', delimiter=',')

if dataset_number == 3:
    # Stored on 'elendil' machine
    image_dataset_path = '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_5_invivo_dynamic/1_1_full_aperture_pw_0/image_dataset.hdf5'
    # Stored on 'ojos3' machine
    image_dataset_path =  '/ojos3/scratch/session1_run5_pw0/image_dataset.hdf5' # Dataset 3

    pennation_angles = np.loadtxt('../../results/sess1_run5_pw0/pennation_angle.csv', delimiter=',')
    theta_1 = np.loadtxt('../../results/sess1_run5_pw0/theta_1.csv', delimiter=',')
    theta_2 = np.loadtxt('../../results/sess1_run5_pw0/theta_2.csv', delimiter=',')
    pos_fas = np.loadtxt('../../results/sess1_run5_pw0/pos_fas.csv', delimiter=',')

# Draw ultrasound images with pennation angle results from the automatic labeling tool and a the ML regressor drawn on it
for cv_num in np.arange(5):

    if dataset_number == 1:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
      
    if dataset_number == 2:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
    
    if dataset_number == 3:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
    
    
    start_frame, end_frame = cv_num*896+1, (cv_num+1)*896
    
    # point is -1 for ds 1 and 3, +1 for ds 2
    if dataset_number in [1,3]:
        point = -1 
    else:
        point = 1
    
    img_loader_obj = ImageLoader(image_dataset_path)
    frame_range = np.arange(start_frame, end_frame+1)
    
    for frame_index in frame_range:
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
    
        with suppress_stdout():
           image_scaled = rescale(log_compress(resize(np.abs(img_data), 
                                               (np.around(np.abs(image_size_z_1-image_size_z_0),4)*1e4, 
                                                np.around(np.abs(image_size_x_1-image_size_x_0),4)*1e4)),
                                        db_range, reference_max = reference_max), scale=1.0, mode='reflect', multichannel=False)
    
        
        img_rgb = plot_double_results(image_scaled, theta_1[frame_index], theta_2[frame_index], 
                                      penn_ANN[frame_index-start_frame],
                                      image_scaled.shape[0]- pos_fas[frame_index], point, 
                                      image_scaled.shape[0]- 60, -1*point, 
                                      frame_index)
    
        cv.imwrite('../../results/frames/frame-%04d'%frame_index + '.png', img_rgb)
