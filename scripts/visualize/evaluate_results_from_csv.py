
import numpy as np
import matplotlib.pyplot as plt
import argparse

# The purspose of this file is to plot and evaluate the .csv files containing the 
# results from the automatic labeling tool

def print_info(arr, name):
    print()
    print('Average ' + name + ': ' + str(np.mean(arr)))
    print('Maximum ' + name + ': ' + str(np.max(arr)))
    print('Minimum ' + name + ': ' + str(np.min(arr)))
    print('90% quantile ' + name + ': ' + str(np.quantile(arr,0.9)))
    print('10% quantile ' + name + ': ' + str(np.quantile(arr,0.1)))
    print('25% quantile ' + name + ': ' + str(np.quantile(arr,0.25)))
    print('75% quantile ' + name + ': ' + str(np.quantile(arr,0.75)))
    print()
    
def plot_features(fr, feats, feature_name, angles):
    plt.figure(figsize=(8,8))
    plt.title(feature_name)
    plt.xlabel('Frame number')
    if angles == True:
        plt.ylabel('Angle (degrees)')
    else:
        plt.ylabel('Position (pixels)')
    plt.plot(fr, feats, linewidth = 1)
    plt.savefig(feature_name + '.png')
    plt.close()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_csv',
        type=str,
        default='',
        help='Path to the file with the .csv functions needed')
    
    parser.add_argument(
        '--path_to_save',
        type=str,
        default='',
        help='Path to for saving the resulting plots')
    
    FLAGS, unparsed = parser.parse_known_args()

    penn =     np.loadtxt(FLAGS.path_to_csv + 'pennation_angle.csv', delimiter=',')
    th1 =      np.loadtxt(FLAGS.path_to_csv + 'theta_1.csv',          delimiter=',')
    th2 =      np.loadtxt(FLAGS.path_to_csv + 'theta_2.csv',          delimiter=',')
    #pos_deep = np.loadtxt(FLAGS.path_to_csv + 'pos_deep.csv',    delimiter=',')
    pos_fas =  np.loadtxt(FLAGS.path_to_csv + 'pos_fas.csv',      delimiter=',')
    
    frames = np.arange(1,len(penn) + 1)
    
    plot_features(frames, penn,     FLAGS.path_to_save + 'pennation_angles', True)
    plot_features(frames, th1,      FLAGS.path_to_save + 'theta_1',          True)
    plot_features(frames, th2,      FLAGS.path_to_save + 'theta_2',          True)
    #plot_features(frames, pos_deep, FLAGS.path_to_save + 'pos_deep',         False)
    plot_features(frames, pos_fas,  FLAGS.path_to_save + 'pos_fas',          False)
    
    print_info(penn, 'pennation angle')
    print_info(th1, 'theta 1')
    print_info(th2, 'theta 2')
    
    count = 0
    differences = []
    for i in range(len(penn)-1):
        diff = np.abs(penn[i] - penn[i+1])
        differences.append(diff)
        if diff>2:
            print(str(diff) + ' degree change between frames ' + str(i+1) + ' and ' + str(i+2))
            count += 1
    print()
    print(str(count)+ ' cases')
    differences = np.array(differences)
    
    print()
    print_info(differences, 'difference in angle')
    
    count = 0
    differences = []
    for i in range(len(pos_fas)-1):
        diff = np.abs(pos_fas[i] - pos_fas[i+1])
        differences.append(diff)
        if diff>10:
            print(str(diff) + ' pixel change between frames ' + str(i+1) + ' and ' + str(i+2))
            count += 1
    print()
    print(str(count)+ ' cases')
    differences = np.array(differences)
    print()
    print_info(differences, 'pixel difference')
     
