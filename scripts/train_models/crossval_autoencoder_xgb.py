import numpy as np

import torch

import time
from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader

from utils.ANN_utils import cross_val_xgb, print_crossval_scores


if __name__ == "__main__":
    
    # Path to a sample dataset. Here, you can provide your own dataset.
    dataset_path = '../../tests/data/rf_dataset.hdf5'
    pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',')
    
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

