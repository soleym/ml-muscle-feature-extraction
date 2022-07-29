import numpy as np
import torch
import joblib

import time
from time import asctime

import sys

# Insert path to pybf library to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

from utils.ANN_utils import AutoEncoderModel
from pybf.pybf.io_interfaces import DataLoader

def predict_ae_xgboost(data, load_path = '../../results/', save_path = None):
    
    
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)  
    
    print('Loading model...')
    print(asctime())
    print()
    
    model = AutoEncoderModel()
    model.load_state_dict(torch.load('../../../results/ae_model.pth'))
    model.eval()

    forest = joblib.load('../../../results/ae_forest.joblib.dat')
    
    reg    = joblib.load('../../../results/ae_regressor.joblib.dat')
    
    print(asctime())
    print()
    
    print('Making predictions...')
    print(asctime())
    print()
    
    X_test = torch.tensor(data.reshape(data.shape[0], 1, data.shape[1], data.shape[2]))
    
    
    compressed = []
    with torch.no_grad():
        for images in X_test:
            images = images.reshape(-1,1, X_test.shape[2], X_test.shape[3])
            outputs, comp = model(images)
            for i in range(len(comp)):
                compr = comp[i].detach().numpy()
                compr = np.reshape(compr, (compr.shape[0]*compr.shape[1]*compr.shape[2]))
                compressed.append(compr)

    compressed = np.array(compressed)

    importances = forest.feature_importances_
    sum_importances = 0
    num_feats = 0
    while sum_importances < 0.99:
        num_feats += 1
        sum_importances = np.sum(np.sort(importances)[-1*num_feats:])

    important_indices = np.argsort(importances)[-1*num_feats:]
    # Remove features of little importance
    feat_trimmed = compressed[:,important_indices]

    penn_predict = reg.predict(feat_trimmed)
    
    if save_path:
        np.savetxt(save_path + 'predicted_ae_xgb_pennation_angle.csv', np.array(penn_predict), delimiter=',')
        
    return penn_predict

if __name__ == "__main__":
    
    # Get the raw data 
    dataset_path = '../../data/rf_dataset.hdf5'
    
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
    
    predict_ae_xgboost(data, load_path = '../../results/', save_path = None)
