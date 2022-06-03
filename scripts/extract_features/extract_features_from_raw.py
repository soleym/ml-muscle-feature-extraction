import numpy as np
from scipy.stats import skew, kurtosis

import time
from time import asctime

import sys
import csv

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
print(path_to_lib)
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces     import DataLoader

#Stored on 'elendil' machine
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

def preprocess(signal):
    gauss = ss.gausspulse(signal,1/(5.2083 * 10 ** 6),1)
    new_data = gauss*signal
    data_hilbert = np.abs(ss.hilbert(new_data))
    preprocessed_data = (np.log10(1 + 0.0003*data_hilbert))
    
    return preprocessed_data


data_loader_obj = DataLoader(dataset_path)
num_frames = data_loader_obj._num_of_frames
num_acquis = data_loader_obj._num_of_acq_per_frame 

print(num_frames)

get_data = True
# Get raw data
if get_data == True:
    print('Getting raw data')
    print()
    data = []
    t1 = time.time()
    for frame in range(num_frames):
        #for acq in range(num_acquis):
        dt = data_loader_obj.get_rf_data(n_frame = frame, m_acq = 0)
        data.append(dt.T)
    print('Time to extract all data in seconds: ' + str(time.time()-t1))
else:
    print('Reuse data')

data = np.array(data)
data = data[:,:800,::2] # Throw away unimportant samples
print()
print('Data shape: ' +str(data.shape))


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
    with open("../../results/ds_features_prepped_w" + str(num_windows) + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(feats)
    print(asctime())


















