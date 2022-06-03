# Take pennation angles from raw data / images, image data and return images with pennation angles drawn on them
import numpy as np
import matplotlib.pyplot as plt

import time
import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
sys.path.insert(0, path_to_lib)

# Insert path to repository, to import utility functions
path_to_utils ='../../'
sys.path.insert(1, path_to_utils)

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


for cv_num in np.arange(5):

    if dataset_number == 1:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
      
    if dataset_number == 2:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
    
    if dataset_number == 3:
        penn_ANN = np.loadtxt('../../results/pennation_angle'+str(cv_num)+'.csv', delimiter=',')
    
    
    start_frame, end_frame = cv_num*896+1, (cv_num+1)*896
    point = -1 # -1 for ds 1 and 3, +1 for ds 2
    
    frame_range = np.arange(start_frame, end_frame+1)
    plt.figure()
    for frame_index in frame_range:
        if frame_index % 50 == 0 or frame_index == 1:
            print(time.asctime() + '    Frame ' + str(frame_index))
    
        # Visualize the pennation angle from the labeling tool and the ML method on the same plot
        plt.figure()
        plt.plot(frame_range,pennation_angles[start_frame:end_frame+1], color = '#00FF00', label = 'ALT')
        plt.plot(frame_range,penn_ANN, color = 'r', label = 'ANN')
        plt.scatter(frame_index, pennation_angles[frame_index], color = 'k')
        plt.scatter(frame_index, penn_ANN[frame_index%896], color = 'k')
        plt.xlabel('Frame index')
        plt.ylabel('Pennation angle (degrees)')
        plt.legend()
        plt.savefig('../../results/frames/a'+str(frame_index)+'.png')
        plt.close()
