"""
   Copyright (C) 2022 ETH Zurich. All rights reserved.

   Author: Soley Hafthorsdottir, ETH Zurich

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

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import torch
import torch.nn as nn

import time
from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader

from utils.ANN_utils import  CNNModel, train_conv_net, compute_scores
                             
def print_crossval_scores(rmse, mse, max_err, mae, r2):
    '''
    Prints the scores from a crossvalidation
    '''
    print('Mean rmse     : ' +str(np.mean(rmse)))
    print('Mean mse      : ' +str(np.mean(mse)))
    print('Mean max_error: ' +str(np.mean(max_err)))
    print('Mean MAE      : ' +str(np.mean(mae)))
    print('Mean r2_score : ' +str(np.mean(r2)))
    print()
    print('rmses     : ' +str(rmse))
    print('mses      : ' +str(mse))
    print('max_errors: ' +str(max_err))
    print('MAEs      : ' +str(mae))
    print('r2_scores : ' +str(r2))
   
if __name__ == "__main__": 
    # Path to a sample dataset. Here, you can provide your own dataset
    dataset_path =  '../../tests/data/rf_dataset.hdf5'
    pennation_angles = np.loadtxt('../../results/pennation_angle.csv', delimiter=',') 
       
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    data_loader_obj = DataLoader(dataset_path)
    # Both n_frame and m_acq start from 0, not 1 (like for image_dataloader)
    num_frames = data_loader_obj._num_of_frames
    num_acquis = data_loader_obj._num_of_acq_per_frame 
    
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
    data = data[:,:800,::2]
    print(data.shape)
    
    penns = np.array(pennation_angles)
    
    def cross_val_cnn(feats, penns, cv=4):
        '''
        Cross validate for given data and split cv (default is 4)
        Inputs are numpy arrays
        '''
        
        b, a = signal.butter(8, 0.04)
        
        all_rmse, all_max_error, all_mse, all_mae, all_r2 = [],[],[],[],[]
        
        step_size = len(penns)//cv
        for i in range(cv):
            print(str(i) + '    ' +str(asctime()))
            X_test = feats[i*step_size:(i+1)*step_size]
            penn_test = penns[i*step_size:(i+1)*step_size]
            X_train = np.delete(feats, np.arange(i*step_size,(i+1)*step_size), axis=0)
            penn_train = np.delete(penns, np.arange(i*step_size,(i+1)*step_size))
            
            if i == 0 or i == cv-1:
                penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
            else:
                # Smooth train before val
                penn_train[np.arange(0,i*step_size)] = signal.filtfilt(b,a, penn_train[np.arange(0,i*step_size)], padlen=150)
                # Smooth train after val
                penn_train[np.arange(i*step_size,len(penn_train))] = signal.filtfilt(b,a, penn_train[np.arange(i*step_size,len(penn_train))], padlen=150)
            
            penn_test = signal.filtfilt(b,a, penn_test, padlen=150)
            
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
            
            y_train = penn_train.reshape(penn_train.shape[0], 1)
            y_test = penn_test.reshape(penn_test.shape[0], 1)
            
            # Standardize the data 
            meanval = np.mean(X_train)
            stdval = np.std(X_train)
            X_train = (X_train - meanval)/stdval
            X_test = (X_test - meanval)/stdval
           
            #X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
            #X_test = (X_test - np.min(X_test))/(np.max(X_test)-np.min(X_test))
     
            X_train = torch.Tensor(X_train)
            y_train = torch.Tensor(y_train.copy())
            X_test = torch.Tensor(X_test)
            y_test = torch.Tensor(y_test.copy())
            
            # batch_size, epoch and iteration
            batch_size = 32
    
            # Pytorch train and test sets
            train = torch.utils.data.TensorDataset(X_train,y_train)
            test = torch.utils.data.TensorDataset(X_test,y_test)
    
            # data loader
            train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
            test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
            
            #Definition of hyperparameters
            num_epochs = 7
    
            # Cross Entropy Loss 
            error = nn.MSELoss()
    
            model = CNNModel(dropout_rate = 0.0)
    
            # Adam Optimizer
            learning_rate = 0.0001
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.0003) 
            
            train_conv_net(model, num_epochs, batch_size, train_loader, optimizer, error)
                
            # Make a prediction for the whole dataset and predict the loss
            print('Predictions for the test data ...')
            predictions = []
            print(asctime())
    
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
                    labels = labels
                    outputs = model(images)
                    for out in range(len(outputs)):
                        predictions.append(outputs[out])
    
            print(asctime())
            penn_predict = np.array(predictions)
            
            y_test = np.array(y_test)
            penn_predict = np.array(penn_predict)
            
            plt.figure(figsize = (8,8))
            plt.scatter(np.delete(np.arange(len(penns)), np.arange(i*step_size,(i+1)*step_size), axis=0), 
                     penn_train, label = 'Train', s = 2)
            
            plt.plot(np.arange(i*step_size,(i+1)*step_size), penn_test, label = 'Test')
            plt.plot(np.arange(i*step_size,(i+1)*step_size),penn_predict, label = 'Predicted pennation angles')
    
            plt.xlabel('Frame')
            plt.ylabel('Pennation_angle')
            plt.legend()
            plt.savefig('cv'+ str(i)+'.png')
            
            np.savetxt('../../results/pennation_angle' +str(i)+'.csv', penn_predict, delimiter=',')
            
            rmse, max_err, mse, mae, r2 = compute_scores(penn_test, penn_predict)
    
            all_rmse.append(rmse)
            all_max_error.append(max_err)
            all_mse.append(mse)
            all_mae.append(mae)
            all_r2.append(r2)
        
        return all_rmse, all_max_error, all_mse, all_mae, all_r2
    
    print('Crossvalidate...')
    print(asctime())
    print()
    rmse, max_err, mse, mae, r2 = cross_val_cnn(data[0:4480], penns[0:4480], cv=5)
    print(asctime())
    print()
    
    print_crossval_scores(rmse, mse, max_err, mae, r2)
    
    
    
    
    
    
    
    
    
    
    
    
