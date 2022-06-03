import numpy as np

import torch

import time
from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import DataLoader

# Insert path to utilities to system path
path_to_util ='../../'
sys.path.insert(1, path_to_util)


from utils.ANN_utils import cross_val_xgb, print_crossval_scores

# Stored on 'elendil' machine
# Session2 run 2 (dataset 1)
#dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session2/20191218_run_2_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'
# Session 1 run 3 (dataset 2)
#dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_3_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'
# Session 1 run 5 (dataset 3)
#dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_5_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'

# Stored on 'ojos3' machine
#dataset_path =  '/ojos3/scratch/session2_run2_pw0/rf_dataset.hdf5' # Dataset 1
#dataset_path =  '/ojos3/scratch/session1_run3_pw0/rf_dataset.hdf5' # Dataset 2
dataset_path =  '/ojos3/scratch/session1_run5_pw0/rf_dataset.hdf5' # Dataset 3

pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',')

#pennation_angles = np.loadtxt('../../results/sess2_run2_pw0/pennation_angle.csv', delimiter=',')
#pennation_angles = np.loadtxt('../../results/sess1_run3_pw0/pennation_angle.csv', delimiter=',')
#pennation_angles = np.loadtxt('../../results/sess1_run5_pw0/pennation_angle.csv', delimiter=',')


if __name__ == "__main__":
    
    
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
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
    print(data.shape)
    
    penns = np.array(pennation_angles)
    print(penns.shape)

    print('Crossvalidate...')
    print(asctime())
    print()
    rmse, max_err, mse, mae, r2 = cross_val_xgb(data[0:4480], penns[:4480], 
                                                cv=5,  
                                                num_epochs = 20, lr = 0.0001, wd = 0.0003,
                                                n_estimators = 100, total_importance = 0.99, save_path = '../../results/')
    print(asctime())
    print()
    
    print_crossval_scores(rmse, mse, max_err, mae, r2)

