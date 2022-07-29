import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from scipy import signal
import pickle

import time
from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader
from utils.ANN_utils import standardize, AutoEncoderModel, train_AE_net

def train_ae_xgb(feats, penns, num_epochs = 7, lr = 0.0005, wd = 0.0003, n_estimators = 100, total_importance = 0.95, model_save_path = None):
    '''
    Train the model, and then save the model
    '''
    print('Training the model...')
    print(asctime())
    print()
    
    b, a = signal.butter(8, 0.04)
    
    penns = signal.filtfilt(b,a, penns, padlen=150)
    
    X_train = feats.reshape(feats.shape[0], 1, feats.shape[1], feats.shape[2])
    
    y_train = penns.reshape(penns.shape[0], 1)
        
    # Standardize the data 
    X_train, _ = standardize(X_train, X_train)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train.copy())
    
    batch_size = 32
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train,y_train)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)    

    #Definition of hyperparameters
    num_epochs = num_epochs
    
    # MSE loss
    criterion = nn.MSELoss()
    
    model = AutoEncoderModel()
    
    # Adam Optimizer
    learning_rate = lr

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = wd)

    model, compressed, loss_list = train_AE_net(model, num_epochs, batch_size, 
                                          train_loader, optimizer, criterion)
    
    # Perform the method with current training and validation
    # Build a forest and compute the impurity-based feature importances
    forest = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 1)

    forest.fit(compressed, penns)

    importances = forest.feature_importances_
    
    sum_importances = 0
    num_feats = 0
    while sum_importances < total_importance:
        num_feats += 1
        sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

    important_indices = np.argsort(importances)[-1*num_feats:]
    # Remove features of little importance
    feat_trimmed = compressed[:,important_indices]
    
    reg = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 1)
    reg.fit(feat_trimmed, penns, verbose = False)
     
          
    if model_save_path:
        print(asctime())
        print('Saving the model to: ', model_save_path)
        
        ae_save = model_save_path + 'ae_model.pth' 
        torch.save(model.state_dict(), ae_save)
        
        forest_save = model_save_path + 'ae_forest.dat'
        pickle.dump(forest, open(forest_save, 'wb'))
        
        print(asctime())
    


if __name__ == "__main__":
    
    #dataset_path = '../../tests/data/rf_dataset.hdf5'
    
    dataset_path = '/scratch/sem21f26/session2_run2_pw0/rf_dataset.hdf5'
    
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
    
    penns = np.array(pennation_angles)

    train_ae_xgb(data, penns, num_epochs = 20, lr = 0.0001, wd = 0.0003,
                 n_estimators = 100, total_importance = 0.99, model_save_path='../../results/')
    
    
    

