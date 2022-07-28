import numpy as np
from scipy.stats import skew, kurtosis
import scipy.signal as ss

import time
from time import asctime

import sys
import csv
import os

# Insert path to pybf library to system path
path_to_lib = '../../'
print(path_to_lib)
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import DataLoader

def extract_features_from_raw_data(path_to_rf_dataset,
                                   path_to_save_features='../../results',
                                   freq_central=5.2083 * 10 ** 6, 
                                   get_data = True):

    def preprocess(signal):
        gauss = ss.gausspulse(signal,1/(freq_central),1)
        new_data = gauss*signal
        data_hilbert = np.abs(ss.hilbert(new_data))
        preprocessed_data = (np.log10(1 + 0.0003*data_hilbert))
        
        return preprocessed_data


    data_loader_obj = DataLoader(path_to_rf_dataset)
    num_frames = data_loader_obj._num_of_frames
    num_acquis = data_loader_obj._num_of_acq_per_frame 

    print('Number of frames in the dataset:', num_frames)

    # Get raw data
    if get_data == True:
        print('Getting raw data')
        print()
        data = []
        t1 = time.time()
        for frame in range(num_frames):
            dt = data_loader_obj.get_rf_data(n_frame = frame, m_acq = 0)
            data.append(dt.T)
        print('Time to extract all data in seconds: ' + str(time.time()-t1))
    else:
        print('Reuse data')

    data = np.array(data)
    data = data[:,:800,::2] # Throw away unimportant samples
    print()
    print('Data shape: ' +str(data.shape))


    # Extract features
    for num_windows in np.array([8]):
        
        window_len = data.shape[1]//num_windows
        print('Window length: ' + str(window_len))
        
        # Extract features from the raw data
        feats = [] 
        t1 = time.time()
        print()
        print('Extracting features')
        print(asctime())
        for dat in range(num_frames): #data
            f1_mean, f2_std, f3_max, f4_min, f5_sumabsdiff, f6_skewness, f7_kurtosis = [],[],[],[],[],[],[]
            proc_dat = preprocess(data[dat])
            
            if dat % 100 == 0:
                print(str(dat) + '    ' +str(asctime()))
            # Divide signal for each channel into x windows of length signal_len / window_len
            for channel in range(data[dat].shape[1]): # channels
                #if channel % 200 == 0:
                #    print('    ' + str(channel))
                for window in range(num_windows): # divide signal into windows
                    win = proc_dat[window*window_len :(window+1)*window_len,channel] 
                    
                    f1_mean.append(np.mean(win))
                    f2_std.append(np.std(win))
                    f3_max.append(np.max(win))
                    f4_min.append(np.min(win))
                    f5_sumabsdiff.append(np.sum(np.abs(np.diff(win))))
                    f6_skewness.append(skew(win))
                    f7_kurtosis.append(kurtosis(win))
                    
            feats.append(np.hstack((f1_mean, f2_std, f3_max, f4_min,f5_sumabsdiff,f6_skewness,f7_kurtosis)))
            
        print(asctime())
        print('Time to extract all data in seconds: ' + str(time.time()-t1))
        
        print()
        print('Saving data')

        # Create folder if it does not exist
        if not os.path.exists(path_to_save_features):
            # Create a new directory because it does not exist 
            os.makedirs(path_to_save_features)

        with open(path_to_save_features + "/ds_features_prepped_w" + str(num_windows) + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(feats)
        print('Finished.')

    return

if __name__ == "__main__":

    rf_dataset_path =  '../../tests/data/rf_dataset.hdf5'

    extract_features_from_raw_data(rf_dataset_path, 
                                   path_to_save_features='../../results')


















