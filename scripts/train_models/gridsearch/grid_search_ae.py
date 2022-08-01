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

import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error

import time

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader

from utils.ANN_utils import AutoEncoderModel, \
                            train_AE_net, standardize

# Path to a sample dataset. Here, you can provide your own dataset
dataset_path = dataset_path = '../../../tests/rf_dataset.hdf5'
pennation_angles = np.loadtxt('../../../results/pennation_angle.csv', delimiter=',')

if __name__ == '__main__':
    start = time.time()

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    print('Getting raw data')
    print()
    # Get and prepare data
    data_loader_obj = DataLoader(dataset_path)
    # Both n_frame and m_acq start from 0, not 1 (like for image_dataloader)
    num_frames = data_loader_obj._num_of_frames
    num_acquis = data_loader_obj._num_of_acq_per_frame 
    
    data = []
    t1 = time.time()
    for frame in range(num_frames):
        for acq in range(num_acquis):
            dt = data_loader_obj.get_rf_data(n_frame = frame, m_acq = acq)
            data.append(dt.T)
    print('Time to extract training data in seconds: ' + str(time.time()-t1))
        
    data = np.array(data)
    data = data[:,:800,:] # Throw away unimportant samples
    
    penns = np.array(pennation_angles)
    
    X_train = data[0:3200]
    y_train = penns[0:3200]
    
    X_test = data[3200:4480]
    y_test = penns[3200:4480]
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
        
        
    # Standardize the data 
    X_train, X_test = standardize(X_train, X_test)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train,y_train)
    test = torch.utils.data.TensorDataset(X_test,y_test)
    
    # construct meshgrid
    num_epochs = [1]#[6,7,8]
    batch_size = [16,32]
    learning_rate = [0.001, 0.0005,0.0001]
    
    space = []
    for ep in range(len(num_epochs)):
        for bs in range(len(batch_size)):
            for lr in range(len(learning_rate)):
                space.append([num_epochs[ep], batch_size[bs], learning_rate[lr]])

    num_iterations = len(num_epochs) * len(batch_size)*  len(learning_rate)
    
    
    # iterate over parameter space
    for iteration in np.arange(num_iterations):
        startit = time.time()
        param_space = space[iteration]
        print('Iteration: ' + str(iteration+1) +' / ' + str(num_iterations))
        print('Parameter Space:')
        print('Num_epochs, batch_size, learning_rate, weight_decay, dropout_rate')
        print(param_space)

        ep = param_space[0]
        bs = param_space[1]
        lr = param_space[2]
        
        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size = bs, shuffle = False)

        error = nn.MSELoss()
        model = AutoEncoderModel()
        
        # Adam Optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[])
        # AdamW Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            
        #Model training
        model, compressed, loss_list = train_AE_net(model, ep, bs, 
                                                    train_loader, optimizer, error)
        
        print()
        print('Predictions for the test data ...')
        predictions = []
        print(time.asctime())
        
        with torch.no_grad():
            for images, labels in test_loader:
                
                images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
                labels = labels
                outputs, comp = model(images)
                for i in range(len(outputs)):
                    predictions.append(outputs[i])
                
        print(time.asctime())
        print()
        
        predictions = np.array(predictions)
        print('Test MSE: ' + str(mean_squared_error(X_test, predictions)))
        
        print()
        
        endit = time.time()
        print('Total time of iteration:')
        print(str(endit-startit) + 'sec')
        print('----------------------------------------------------------------------------'
              '--------------------------------------------------')
    end = time.time()
    print('Total time:')
    print(str(end - start) + ' sec')
