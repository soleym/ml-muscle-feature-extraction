import matplotlib
import matplotlib.pyplot as plt
import sys

# Insert path to pybf library to system path
path_to_lib ='/home/sem21f26'
print(path_to_lib)
sys.path.insert(0, path_to_lib)

from pybf.pybf.io_interfaces import DataLoader

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

# Get one frame
n_frame = 1
data_loader_obj = DataLoader(dataset_path)
dt = data_loader_obj.get_rf_data(n_frame = n_frame, m_acq = 0)

data_trimmed = dt.T[:800,::2]
#print(data_trimmed.shape)

from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgcykmr')

# Plot the first four channels of the frame
plt.plot()
plt.plot(data_trimmed[:,:4])
plt.legend(['Channel 1','Channel 2','Channel 3','Channel 4'],
           bbox_to_anchor=(1.05, 1))#,
          #labelcolor = ['red','orange','yellow','green','cyan','blue','purple','magenta'])
plt.xlabel('Samples')
plt.ylabel('Signal')
plt.grid()
plt.savefig('rawchannels_frame'+str(n_frame))
