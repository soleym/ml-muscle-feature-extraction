import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy import signal

import xgboost as xgb

import time

import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import DataLoader

# Insert path to utilities to system path
path_to_util ='../../../'
sys.path.insert(1, path_to_util)

from utils.ANN_utils import AutoEncoderModel, \
                            train_AE_net, standardize, compute_scores

# Stored on 'elendil' machine
# Session2 run 2 (dataset 1)
#dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session2/20191218_run_2_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'
# Session 1 run 3 (dataset 2)
#dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_3_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'
# Session 1 run 5 (dataset 3)
dataset_path =  '/usr/scratch2/elendil/vsergei/Measurements-EPFL-201912_processed/20191218 session1/20191218_run_5_invivo_dynamic/1_1_full_aperture_pw_0/rf_dataset.hdf5'

pennation_angles = np.loadtxt('../../../results/pennation_angle.csv', delimiter=',')

#pennation_angles = np.loadtxt('/home/sem21f26/visualize/ALT/new_method_results/sess2_run2_pw0/pennation_angle.csv', delimiter=',')
#pennation_angles = np.loadtxt('/home/sem21f26/visualize/ALT/new_method_results/sess1_run3_pw0/pennation_angle.csv', delimiter=',')
#pennation_angles = np.loadtxt('/home/sem21f26/visualize/ALT/new_method_results/sess1_run5_pw0/pennation_angle.csv', delimiter=',')

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
    data = data[:,:800,::2] # Throw away unimportant samples
    
    penns = np.array(pennation_angles)
    b, a = signal.butter(8, 0.04)
    
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
    
    #X_train = (X_train - np.min(X_train))/(np.max(X_train)-np.min(X_train))
    #X_test = (X_test - np.min(X_test))/(np.max(X_test)-np.min(X_test))

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train.copy())
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test.copy())
    
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train,y_train)
    test = torch.utils.data.TensorDataset(X_test,y_test)
    
    # construct meshgrid
    # Autoencoder parameters
    num_epochs = [7]
    batch_size = [32,64]
    learning_rate = [0.0001,0.0003,0.0005]
    weight_decay = [0.0001,0.0003,0.0005]
    
    # xgboost parameters
    n_estim = [100]
    total_imp = [0.98,0.99]
    
    
    ae_space = []
    xgb_space = []
    imp_space = []
    for ep in range(len(num_epochs)):
        for bs in range(len(batch_size)):
            for lr in range(len(learning_rate)):
                for wd in range(len(weight_decay)):
                    ae_space.append([num_epochs[ep], batch_size[bs], learning_rate[lr], weight_decay[wd]])
    for n_est in range(len(n_estim)):
        xgb_space.append([n_estim[n_est]])    
    for tot_imp in range(len(total_imp)):
        imp_space.append([total_imp[tot_imp]])

    num_ae_iterations  = len(num_epochs) * len(batch_size)*  len(learning_rate) * len(weight_decay)
    num_xgb_iterations = len(n_estim)
    num_imp_iterations = len(total_imp)
    
    # results = np.empty(shape=np.array(space[0]).shape)
    # iterate over parameter space
    for ae_iteration in np.arange(num_ae_iterations):
        startit = time.time()
        
        ae_param_space = ae_space[ae_iteration]

        ep = ae_param_space[0]
        bs = ae_param_space[1]
        lr = ae_param_space[2]
        wd = ae_param_space[3]
        
        # Pytorch train and test sets
        train = torch.utils.data.TensorDataset(X_train,y_train)
        test = torch.utils.data.TensorDataset(X_test,y_test)
        
        # data loader
        train_loader = torch.utils.data.DataLoader(train, batch_size = bs, shuffle = False)
        test_loader = torch.utils.data.DataLoader(test, batch_size = bs, shuffle = False)
    
    
        #Definition of hyperparameters
        num_epochs = ep
        
        # MSE loss
        criterion = nn.MSELoss()
        
        model = AutoEncoderModel()
        
        # AdamW Optimizer
        learning_rate = lr
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wd)
        
        model, compressed_train, loss_list = train_AE_net(model, num_epochs, batch_size, 
                                              train_loader, optimizer, criterion)
        
        
        
        print()
        print('Predictions for the test data ...')
        compressed = []
        print(time.asctime())
        
        counter = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
                labels = labels
                outputs, comp = model(images)
                for i in range(len(comp)):
                    compr = comp[i].detach().numpy()
                    compr = np.reshape(compr, (compr.shape[0]*compr.shape[1]*compr.shape[2]))
                    compressed.append(compr)
                counter += 1
    
        print(time.asctime())
        compressed_test = np.array(compressed)
        #print(compressed_test.shape)
        print()
        
        for xgb_iteration in np.arange(num_xgb_iterations):
            start_xgb = time.time()       
            xgb_param_space = xgb_space[xgb_iteration]            
            n_est = xgb_param_space[0]
            
            # Perform the method with current training and validation
            # Build a forest and compute the impurity-based feature importances
            forest = xgb.XGBRegressor(n_estimators=n_est, random_state=0, n_jobs = -1, verbosity = 1)
    
            forest.fit(compressed_train, penns[0:3200])
    
            importances = forest.feature_importances_

            for imp_iteration in np.arange(num_imp_iterations):

                imp_param_space = imp_space[imp_iteration]

                print('Iteration: ' + str(ae_iteration+1) +'/' + str(num_ae_iterations)+'    ' +str(xgb_iteration + 1) +'/' + str(num_xgb_iterations) + '    ' + str(imp_iteration + 1) +'/' + str(num_imp_iterations))
                print('Parameter Space:')
                print('Num_epochs, batch_size, learning_rate')
                print(ae_param_space) 
                print('n_estimators')
                print(xgb_param_space)
                print('total_importance')
                print(imp_param_space)

                tot_imp = imp_param_space[0]
                
                sum_importances = 0
                num_feats = 0
                while sum_importances < tot_imp:
                    num_feats += 1
                    sum_importances = np.sum(np.sort(importances)[-1*num_feats:])
    
                important_indices = np.argsort(importances)[-1*num_feats:]
                print('Number of features used: ' + str(len(important_indices)))
                print()

                # Remove features of little importance
                feat_train_trimmed = compressed_train[:,important_indices]
                feat_test_trimmed = compressed_test[:,important_indices]
            
                # Perform regression on selected features with a new xgb regressor
                # We can use a different regressor here
                reg = xgb.XGBRegressor(n_estimators=n_est, random_state=0, n_jobs = -1, verbosity = 1)
                reg.fit(feat_train_trimmed, penns[0:3200],verbose = False)
                   #eval_set = [(feat_train_trimmed, y_train),(feat_test_trimmed, y_test)],
                   #eval_metric = ['rmse', 'mae'],
            
                # Predict pennation angles and get scores
                penn_predict = reg.predict(feat_test_trimmed)
            
                rmse, max_err, mse, mae, r2 = compute_scores(penns[3200:4480], penn_predict) 
    
                print('----------------------------------------------------------------------------'
                      '--------------------------------------------------')
            print('Total time of (XGB) iteration:')
            print(str(time.time()-start_xgb) + 'sec')
            print('----------------------------------------------------------------------------'
                  '--------------------------------------------------')
        endit = time.time()
        print('Total time of (AE) iteration:')
        print(str(endit-startit) + 'sec')
        print('----------------------------------------------------------------------------'
                  '--------------------------------------------------')
    end = time.time()
    print('Total time:')
    print(str(end - start) + ' sec')
