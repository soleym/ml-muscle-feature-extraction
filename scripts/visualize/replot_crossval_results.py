# Replot with better font sizes
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse
import matplotlib

font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 18
       }

matplotlib.rc('font', **font)

#ds_num = 1 # 1: sess2_run2, 2: sess1_run3, 3: sess1_run5
#cv_num = 5 # cv_num - fold crossvalidation

parser = argparse.ArgumentParser()

parser.add_argument(
    '--ds_num',
    type = int,
    default = 1,
    help = 'dataset number, 1, 2 or 3'
)

parser.add_argument(
    '--cv_num',
    type = int,
    default = 5,
    help = 'crossvalidation number, i.e. cv_num-fold crossvalidation'
)

FLAGS, unparsed = parser.parse_known_args()

# Load GT pennation angles
if FLAGS.ds_num == 1:
    GT_path = '/home/sem21f26/visualize/ALT/new_method_results/sess2_run2_pw0/pennation_angle.csv'
if FLAGS.ds_num == 2:
    GT_path = '/home/sem21f26/visualize/ALT/new_method_results/sess1_run3_pw0/pennation_angle.csv'
if FLAGS.ds_num == 3:
    GT_path = '/home/sem21f26/visualize/ALT/new_method_results/sess1_run5_pw0/pennation_angle.csv'

penns = np.loadtxt(GT_path, delimiter=',')
penns = penns[:4480]

# Get smoothing filter
b, a = signal.butter(8, 0.04)

# Plot the figure
plt.figure(figsize = (8,8))
step_size = len(penns)//FLAGS.cv_num  
for cv_index in range(FLAGS.cv_num):
    penn_test = penns[cv_index*step_size:(cv_index+1)*step_size]    
    penn_train = np.delete(penns, np.arange(cv_index*step_size,(cv_index+1)*step_size))
    
    penn_test = signal.filtfilt(b,a, penn_test, padlen=150)
    if cv_index == 0 or cv_index == 5-1:
        penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
    else:
        # Smooth train before val
        penn_train[np.arange(0,cv_index*step_size)] = signal.filtfilt(b,a, penn_train[np.arange(0,cv_index*step_size)], padlen=150)
        # Smooth train after val
        penn_train[np.arange(cv_index*step_size,len(penn_train))] = signal.filtfilt(b,a, penn_train[np.arange(cv_index*step_size,len(penn_train))], padlen=150)
        
    if FLAGS.ds_num == 1:
        penn_predict = np.loadtxt('/home/sem21f26/visualize/xgboost/00currentbest/ae_xgb/cv5_AE_stin_linout_sess2/pennation_angle'+str(cv_index)+'.csv', delimiter=',')
    if FLAGS.ds_num == 2:
        penn_predict = np.loadtxt('/home/sem21f26/visualize/xgboost/00currentbest/ae_xgb/cv5_ep20_sess1_3/pennation_angle'+str(cv_index)+'.csv', delimiter=',')
    if FLAGS.ds_num == 3:
        penn_predict = np.loadtxt('/home/sem21f26/visualize/xgboost/00currentbest/ae_xgb/cv5_ep20_sess1_5/pennation_angle'+str(cv_index)+'.csv', delimiter=',')
    
    if cv_index == 0:
        plt.plot(np.arange(cv_index*step_size,(cv_index+1)*step_size), penn_test, 
                 label = 'True angles', c = 'k')
    else:
        plt.plot(np.arange(cv_index*step_size,(cv_index+1)*step_size), penn_test, c = 'k')
        
    plt.plot(np.arange(cv_index*step_size,(cv_index+1)*step_size),penn_predict, 
             label = 'Prediction '+str(cv_index+1))

        
    plt.scatter(np.delete(np.arange(len(penns)), np.arange(cv_index*step_size,(cv_index+1)*step_size), axis=0), 
             penn_train, s = 2,c = 'k')

savestring = f'/home/sem21f26/visualize/fig6_visualizations/dataset{FLAGS.ds_num}.png'
plt.xlabel('Frame')
plt.ylabel('Pennation angle (degrees)')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig(savestring)
