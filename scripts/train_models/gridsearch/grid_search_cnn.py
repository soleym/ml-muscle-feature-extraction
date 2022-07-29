import numpy as np

import torch
import torch.nn as nn

import time
from time import asctime

from scipy import signal

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import DataLoader

from utils.ANN_utils import CNNModel, \
                            train_conv_net, standardize, \
                            print_scores, plot_loss, plot_prediction

if __name__ == '__main__':
    
    dataset_path = '../../../tests/data/rf_dataset.hdf5'
    pennation_angles = np.loadtxt('../../../results/pennation_angle.csv', delimiter=',')

    
    start = time.time()

    b, a = signal.butter(8, 0.04)

    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    
    
    # Get and prepare data
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
    data = data[:,:800,::2] # Throw away unimportant samples
    
    penns = np.array(pennation_angles)
    
    X_train = data[0:3200]
    y_train = penns[0:3200]
    
    X_test = data[3200:3200+1280]
    y_test = penns[3200:3200+1280]    
    
    y_train = signal.filtfilt(b,a, y_train, padlen=150) 
    y_test = signal.filtfilt(b,a, y_test, padlen=150) 
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
        
        
    # Standardize the data 
    X_train, X_test = standardize(X_train, X_test)
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train.copy())
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test.copy())
    
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train,y_train)
    test = torch.utils.data.TensorDataset(X_test,y_test)
    
    # construct meshgrid
    num_epochs = [50]
    batch_size = [32]
    learning_rate = [0.0001] #0.001
    weight_decay  = [0.0003]
    dropout_rate = [0.0]
    
    
    space = []
    for ep in range(len(num_epochs)):
        for bs in range(len(batch_size)):
            for lr in range(len(learning_rate)):
                for wd in range(len(weight_decay)):
                    for dr in range(len(dropout_rate)):
                        space.append([num_epochs[ep], batch_size[bs], 
                                      learning_rate[lr], weight_decay[wd], dropout_rate[dr]])

    num_iterations = len(num_epochs) * len(batch_size)*  len(learning_rate) * len(weight_decay)*len(dropout_rate)
    
    
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
        wd = param_space[3]
        dr = param_space[4]
        
        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size = bs, shuffle = False)

        error = nn.MSELoss()
        model = CNNModel(dr)
        
        # Adam Optimizer
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[])
        # AdamW Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)
            
        #CNN model training
        model, loss_list = train_conv_net(model, ep, bs, 
                                          train_loader, optimizer, error)
        
        # Make a prediction for the whole dataset and predict the loss
        print('Predictions for the test data ...')
        predictions = []
        print(asctime())
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
                labels = labels
                outputs = model(images)
                for i in range(len(outputs)):
                    predictions.append(outputs[i])
            
        print(asctime())
        
        predictions = np.array(predictions)
        print_scores(y_test, predictions)
        plot_loss(loss_list)
        plot_prediction(y_test, predictions)

        endit = time.time()
        print('Total time of iteration:')
        print(str(endit-startit) + 'sec')
        print('----------------------------------------------------------------------------'
              '--------------------------------------------------')
    end = time.time()
    print('Total time:')
    print(str(end - start) + ' sec')
