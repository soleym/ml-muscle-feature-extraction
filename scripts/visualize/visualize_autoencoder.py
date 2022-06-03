import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchsummary import summary

import time
from time import asctime

from scipy import signal


import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import DataLoader

# Insert path to utilities to system path
path_to_util ='../../'
sys.path.insert(1, path_to_util)


from utils.ANN_utils import standardize, AutoEncoderModel, train_AE_net, plot_loss

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
#pennation_angles = np.loadtxt(../../results/sess1_run5_pw0/pennation_angle.csv', delimiter=',')

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
    b, a = signal.butter(8, 0.04)
    #penns = signal.filtfilt(b,a,penns,padlen=150)
    print(penns.shape)
    
    X_train = data[0:3200]
    y_train = signal.filtfilt(b,a, penns[0:3200], padlen=150)
    
    X_test = data[3200:3200+1280]
    y_test = signal.filtfilt(b,a, penns[3200:3200+1280], padlen=150)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
        
        
    # Standardize the data 
    X_train, X_test = standardize(X_train, X_test)
    
    print(np.min(X_train))
    print(np.max(X_train))
    print(np.mean(X_train))

    print(np.min(X_test))
    print(np.max(X_test))
    print(np.mean(X_test))
 
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train.copy())
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test.copy())
    
    batch_size = 32
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train,y_train)
    test = torch.utils.data.TensorDataset(X_test,y_test)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


    #Definition of hyperparameters
    num_epochs = 7
    
    # MSE loss
    criterion = nn.MSELoss()
    
    model = AutoEncoderModel()
    
    # Adam Optimizer
    learning_rate = 0.0001# 0.0005
    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=0.0003)
    
    summary(model, (1,800,96))

    model, train_comp, loss_list = train_AE_net(model, num_epochs, batch_size, 
                                          train_loader, optimizer, criterion)
    
    plot_loss(loss_list)
    

    print('Predictions for the test data ...')
    compressed = []
    print(asctime())
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
            labels = labels
            outputs, comp = model(images)
            for i in range(len(comp)):
                compr = np.array(comp[i])
                compr = np.reshape(compr, (compr.shape[0],compr.shape[1],compr.shape[2]))
                compressed.append(compr)

    print(asctime())
    compressed = np.array(compressed)
    print(compressed.shape)



    #Batch of test images
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    
    #Sample outputs
    output, comp = model(images)
    images = images.numpy()
    print(images.shape)
    
    print(comp.shape)
    
    output = output.view(batch_size, 1, 800, 96)
    output = output.detach().numpy()
    
    print(output.shape)
    
    print('Original Images')
    for i in range(96):
        #plt.imshow(images[i][0].T, cmap = 'gray')
        plt.plot(images[0,0,:,i]) #bs, 
        #plt.xticks = ([])
        #plt.yticks = ([])
        plt.xlabel('Sample')
        plt.ylabel('Signal')
        plt.savefig('original'+str(i)+'.png')
        #ax.set_title(labels[idx])
        plt.close()
    
    print('Reconstructed Images')
    for i in range(96):
        #plt.imshow(output[i][0].T, cmap = 'gray')
        plt.plot(output[0,0,:,i])
        #plt.xticks = ([])
        #plt.yticks = ([])
        plt.xlabel('Sample')
        plt.ylabel('Signal')
        #ax.set_title(labels[idx])
        plt.savefig('recon'+ str(i) +'.png')
        plt.close()
