import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from scipy import signal

import xgboost as xgb


from time import asctime

class CNNModel(nn.Module): 
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2 + batch norm
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.cnn2_bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Fully connected 1 and 2
        self.fc1 = nn.Linear(32 *198 *22, 128) #22,46
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # Set 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn2_bn(out)
        out = self.dropout(out)
        
        #Flatten
        out = out.view(out.size(0), -1)

        #Dense
        out = self.fc1(out)
        out = self.relu3(out)
        #out = self.fc1_bn(out)
        
        out = self.fc2(out)
        
        return out
    
class AutoEncoderModel(nn.Module): 
    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        
        #Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.batchnorm = nn.BatchNorm2d(32)
        
        # Middle
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(64, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 8, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    
    def forward(self, x):
        #Encoder
        x = F.leaky_relu(self.conv1(x))
        x = self.maxpool(x)
        out1 = x
        
        x = F.leaky_relu(self.conv2(x))
        x = self.maxpool(x)
        out2 = x        

        x = F.leaky_relu(self.conv3(x))
        x_comp = self.batchnorm(self.maxpool(x))
        out3 = x_comp        

        # Middle
        x = F.leaky_relu(self.conv4(x_comp))
        x = self.batchnorm(F.leaky_relu(self.conv5(x)))
        
        # Decoder with skip connections
        x = F.leaky_relu(self.t_conv1(torch.cat((x,out3),1)))
        x = F.leaky_relu(self.t_conv2(torch.cat((x,out2),1)))
        x = self.t_conv3(torch.cat((x,out1),1))

        return x, x_comp

    
def train_conv_net(model, num_epochs, batch_size, train_loader, optimizer, error): # Training our network
    '''
    Trains a CNN model, given:
        num_epochs: the number of epochs
        batch_Size: the batch size
        train_loader: the train loader
        optimizer: the optimizer
        error: the loss function
        
    Returns:
        model: The trained model
        loss_list: A list containing the training loss for each epoch
    '''

    loss_list = []
    print('Start: ' + str(asctime()))
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            train = torch.autograd.Variable(images.view(batch_size,1,images.shape[2],images.shape[3]))
            labels = torch.autograd.Variable(labels)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(train)
            # Calculate softmax and mse loss
            loss = error(outputs, labels)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
        # store loss and iteration
        loss_list.append(loss.data)
    
        # Print Time and loss
        print('  ' + str(asctime()) + '    Epoch: {}  Loss: {} '.format(epoch, loss.data))
    
    return model, loss_list

def train_AE_net(model, num_epochs, batch_size, train_loader, optimizer, error):
    '''
    Trains an Autoencoder CNN model, given:
        num_epochs: the number of epochs
        batch_Size: the batch size
        train_loader: the train loader
        optimizer: the optimizer
        error: the loss function
        
    Returns:
        model: The trained model
        comp: compressed features from encoder
        loss_list: A list containing the training loss for each epoch
    '''
    loss_list = []
    print('Start: ' + str(asctime()))
    for epoch in range(1, num_epochs+1):
        # monitor training loss
        train_loss = 0.0
    
        #Training
        compressed = []
        for data in train_loader:
            images, _ = data
            images = images
            optimizer.zero_grad()
            outputs, comp = model(images)
            if epoch == num_epochs:
                for i in range(len(comp)):
                    compr = comp[i].detach().numpy()
                    compr = np.reshape(compr, (compr.shape[0]*compr.shape[1]*compr.shape[2]))
                    compressed.append(compr)
            
            loss = error(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
            
        compressed = np.array(compressed)
              
        train_loss = train_loss/len(train_loader)
        loss_list.append(train_loss)
        print('  ' + str(asctime()) + '    epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return model, compressed, loss_list
        
        
def print_scores(y_true, y_pred):
    '''
    Print the following regression scores given ground truth and predictions:
        Mean squared error (MSE), 
        Mean absolute error (MAE),
        Maximum error
        R2 score
    '''
    print('MSE      : ' + str(mean_squared_error(y_true, y_pred)))
    print('MAE      : ' + str(mean_absolute_error(y_true, y_pred)))
    print('Max error: ' + str(max_error(y_true, y_pred)))
    print('R2 score : ' + str(r2_score(y_true, y_pred)))
    
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
    
def plot_prediction(y_true, y_pred):
    '''
    Plot the predictions and ground truth 
    '''
    plt.plot(y_true, label = 'Ground truth')
    plt.plot(y_pred, label = 'Predictions')
    plt.xlabel('Frame')
    plt.ylabel('Pennation angle')
    plt.legend()
    plt.savefig('predictions.png')
    plt.show()
    plt.close()
    
def plot_loss(loss_list):
    '''
    Plot the loss function given a list of losses for each epoch
    '''
    plt.plot(np.arange(len(loss_list)),loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.savefig('loss.png')
    plt.show()
    plt.close()
    
def standardize(X_train, X_test):
    '''
    Standardize the data based on the training data statistics
    '''
    meanval = np.mean(X_train)
    stdval = np.std(X_train)

    X_train = (X_train - meanval)/stdval
    X_test = (X_test - meanval)/stdval
    
    return X_train, X_test

def cross_val_xgb(feats, penns, cv=4, num_epochs = 7, lr = 0.0005, wd = 0.0003, n_estimators = 100, total_importance = 0.95, save_path = ''):
    '''
    Cross validate the pca method for given data and split cv (default is 4)
    '''
    all_rmse, all_max_error, all_mse, all_mae, all_r2 = [],[],[],[],[]
    
    b, a = signal.butter(8, 0.04)
    
    step_size = len(penns)//cv
    for cv_index in range(cv):
        print(str(cv_index) + '    ' +str(asctime()))
        feat_test = feats[cv_index*step_size:(cv_index+1)*step_size]
        penn_test = penns[cv_index*step_size:(cv_index+1)*step_size]
        feat_train = np.delete(feats, np.arange(cv_index*step_size,(cv_index+1)*step_size), axis=0)
        penn_train = np.delete(penns, np.arange(cv_index*step_size,(cv_index+1)*step_size))
        
        #penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
        if cv_index == 0 or cv_index == cv-1:
            penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
        else:
            # Smooth train before val
            penn_train[np.arange(0,cv_index*step_size)] = signal.filtfilt(b,a, penn_train[np.arange(0,cv_index*step_size)], padlen=150)
            # Smooth train after val
            penn_train[np.arange(cv_index*step_size,len(penn_train))] = signal.filtfilt(b,a, penn_train[np.arange(cv_index*step_size,len(penn_train))], padlen=150)

        
        penn_test = signal.filtfilt(b,a, penn_test, padlen=150)
        
        
        
        X_train = feat_train.reshape(feat_train.shape[0], 1, feat_train.shape[1], feat_train.shape[2])
        X_test = feat_test.reshape(feat_test.shape[0], 1, feat_test.shape[1], feat_test.shape[2])
        
        y_train = penn_train.reshape(penn_train.shape[0], 1)
        y_test = penn_test.reshape(penn_test.shape[0], 1)
            
            
        # Standardize the data 
        X_train, X_test = standardize(X_train, X_test)
        
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
        num_epochs = num_epochs
        
        # MSE loss
        criterion = nn.MSELoss()
        
        model = AutoEncoderModel()
        
        # Adam Optimizer
        learning_rate = lr

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = wd)
        #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        
        model, compressed_train, loss_list = train_AE_net(model, num_epochs, batch_size, 
                                              train_loader, optimizer, criterion)
        
        
        torch.save(model.state_dict(), 'model'+str(cv_index)+'.pth')
        print('Predictions for the test data ...')
        compressed = []
        print(asctime())
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
                labels = labels
                outputs, comp = model(images)
                for i in range(len(comp)):
                    compr = comp[i].detach().numpy()
                    compr = np.reshape(compr, (compr.shape[0]*compr.shape[1]*compr.shape[2]))
                    compressed.append(compr)
    
        print(asctime())
        compressed_test = np.array(compressed)
        print(compressed_test.shape)
        
        # Perform the method with current training and validation
        # Build a forest and compute the impurity-based feature importances
        forest = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 1)

        forest.fit(compressed_train,penn_train)

        importances = forest.feature_importances_
        
        sum_importances = 0
        num_feats = 0
        while sum_importances < total_importance:
            num_feats += 1
            sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

        important_indices = np.argsort(importances)[-1*num_feats:]
        # Remove features of little importance
        feat_train_trimmed = compressed_train[:,important_indices]
        feat_test_trimmed = compressed_test[:,important_indices]
        
        reg = xgb.XGBRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbosity = 1)
        reg.fit(feat_train_trimmed, penn_train, verbose = False)
        
        # Predict pennation angles and get scores
        penn_predict = reg.predict(feat_test_trimmed)
        
        plt.figure(figsize = (8,8))
        plt.scatter(np.delete(np.arange(len(penns)), np.arange(cv_index*step_size,(cv_index+1)*step_size), axis=0), 
                 penn_train, label = 'Train', s = 2)
        
        plt.plot(np.arange(cv_index*step_size,(cv_index+1)*step_size), penn_test, label = 'Test')
        plt.plot(np.arange(cv_index*step_size,(cv_index+1)*step_size),penn_predict, label = 'Predicted pennation angles')

        plt.xlabel('Frame')
        plt.ylabel('Pennation_angle')
        plt.legend()
        plt.savefig('cv'+ str(cv_index)+'.png')
        
        np.savetxt(save_path + 'pennation_angle' +str(cv_index)+'.csv', penn_predict, delimiter=',')
        

        rmse, max_err, mse, mae, r2 = compute_scores(penn_test, penn_predict) 

        all_rmse.append(rmse)
        all_max_error.append(max_err)
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
    
    return all_rmse, all_max_error, all_mse, all_mae, all_r2

def compute_scores(penn_true, penn_pred, printout = True):
    '''
    Compute regression scores predicted pennation angles and print the scores 
    if printout is true
    '''
    rmse = np.sqrt(np.mean(np.sum((penn_pred - penn_true)**2))/len(penn_true))
    max_err = max_error(penn_true, penn_pred)
    mse = mean_squared_error(penn_true, penn_pred)
    mae = mean_absolute_error(penn_true, penn_pred)
    r2 = r2_score(penn_true, penn_pred)

    if printout == True:
        print()
        print('RMSE      : ' + str(rmse))
        print('MSE       : ' + str(mse))
        print('MAE       : ' + str(mae))
        print('max_error : ' + str(max_err))
        print('R2_score  : ' + str(r2))
        print()
    
    return rmse, max_err, mse, mae, r2
    
    
    
    
    
    
    
    
    
    
    
    
    

    
