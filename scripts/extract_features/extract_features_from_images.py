import numpy as np
import cv2 as cv

from skimage import feature
from skimage.transform import radon, rescale, resize

import time
import sys
import os

# Insert path to pybf library and utility functions to system path
path_to_lib_and_utils ='../../'
sys.path.insert(0, path_to_lib_and_utils)

from pybf.pybf.io_interfaces import ImageLoader
from pybf.pybf.visualization import log_compress

from utils.image_labeling import weight_angle,get_reference_max, get_peaks, plot_results, filter_canny_aponeuroses

# Useful code to suppress output, from:  ######################################
# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
from contextlib import contextmanager
import os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
###############################################################################
# More useful code to suppress warnings from miniconda, from
# https://stackoverflow.com/questions/61444572/ignore-all-warnings-from-a-module          
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
###############################################################################

weighted_pos_fas = 0
weighted_ang_fas = 0

num_peaks_apo = 45 # was 35 
num_peaks_fas = 40

def extract_features_from_images(path_to_dataset, path_to_save_results='../../results', save_images=True):

    th1 = []
    th2 = []
    penn = []
    pos_deep = []
    pos_fascicle = []

    # Initate image loader
    img_loader_obj = ImageLoader(image_dataset_path)

    # Create folder for frames if needed
    if save_images and (not os.path.exists(path_to_save_results)):
        # Create a new directory because it does not exist 
        os.makedirs(path_to_save_results + '/frames')

    # Iterate over images and process them
    for frame_index in np.arange(0,len(img_loader_obj.frame_indices)):
        if frame_index % 50 == 0 or frame_index == 1:
            print(time.asctime() + '    Frame ' + str(frame_index))

        img_data = img_loader_obj.get_high_res_image(frame_index)

        # Calculate image sizes
        pixels_coords = img_loader_obj.get_pixels_coords()
        image_size_x_0 = pixels_coords[0, :].min()
        image_size_x_1 = pixels_coords[0, :].max()
        image_size_z_0 = pixels_coords[1, :].min()
        image_size_z_1 = pixels_coords[1, :].max()

        attenuate = True
        att = ''
        reference_max = None
        if attenuate == True:
            att = 'with saturated maximums '
            reference_max = get_reference_max(img_data, 0.2)

        db_range = 50

        with suppress_stdout():
            image_scaled = rescale(log_compress(resize(np.abs(img_data), 
                                                (np.around(np.abs(image_size_z_1-image_size_z_0),4)*1e4, 
                                                np.around(np.abs(image_size_x_1-image_size_x_0),4)*1e4)),
                                                db_range, reference_max = reference_max), scale=1.0, mode='reflect', multichannel=False)

        # Use Radon transform on the canny edge map with sigma = 6.5 (to detect aponeuroses)
        sigma = 6.5
        edges = feature.canny(image_scaled, sigma = sigma)
        theta = np.linspace(0., 180., max(edges.shape), endpoint=False)
        sinogram = radon(edges, theta=theta, circle=True)
        dx, dy = 0.5 * 180.0 / max(image_scaled.shape), 0.5 / sinogram.shape[0]
        
        
        # Extract peak points
        x, y = sinogram.shape
        top_idxs = get_peaks(sinogram, num_peaks_apo)

        # deep aponeurosis
        pos1 = []
        ang1 = []
        weights1 = []
        # superficial aponeurosis
        pos2 = []
        ang2 = []
        weights2 = []
        
        # Separate aponeuroses into deep and superficial
        for i in range(len(top_idxs)):
            if top_idxs[i][0] < x/2: # Deep aponeurosis
                pos1.append(top_idxs[i][0])
                ang1.append(theta[top_idxs[i][1]])
                weights1.append(sinogram[top_idxs[i][0]][top_idxs[i][1]-1])
            if top_idxs[i][0] > (2*x)/3: # Superficial aponeurosis
                pos2.append(top_idxs[i,0])
                ang2.append(theta[top_idxs[i,1]])
                weights2.append(sinogram[top_idxs[i][0]][top_idxs[i][1]-1])
                
        # Filter outliers in aponeuroses (remove peaks that point in a different direction than the majortiy)
        # Also remove points that are too close to the center
        ang1, pos1, weights1 = filter_canny_aponeuroses(ang1,pos1,weights1)
        ang2, pos2, weights2 = filter_canny_aponeuroses(ang2,pos2,weights2)


        # Look for fasicles only between max(pos1) and min(pos2)
        # Extract peak points for the fascicles
        sinogram = radon(image_scaled, theta=theta, circle=True)
        subtracted_sinogram = sinogram - np.mean(image_scaled)
        
        #Get gradients and plot them
        gradients = np.gradient(subtracted_sinogram)
        Grad1 = np.array(gradients[0])
        Grad2 = np.array(gradients[1])
        
        #if frame_index == 0:
        lower_bound_1 = max(pos1)+20
        upper_bound_1 = min(pos2)-20
        
        lower_bound_2 = 0
        upper_bound_2 = Grad1.shape[1]
        
        '''
        else:
            last_pos = sinogram.shape[0]-weighted_pos_fas
            lower_bound_1 = last_pos - 10
            upper_bound_1 = last_pos + 11
            
            for i in range(len(theta)):
                if np.abs(theta[i] - weighted_ang_fas) < dx:
                    last_angle = i
            
            lower_bound_2 = last_angle-10
            upper_bound_2 = last_angle+11
        ''' 
        G1_slice = Grad1[lower_bound_1:upper_bound_1,lower_bound_2:upper_bound_2]
        G2_slice = Grad2[lower_bound_1:upper_bound_1,lower_bound_2:upper_bound_2]
            
        # Extract peak points
        x, y = G1_slice.shape 

        top_idxs_G1 = get_peaks(G1_slice, num_peaks_fas)
        top_idxs_G2 = get_peaks(G2_slice, num_peaks_fas)
        
        for i in range(len(top_idxs_G1)):
            top_idxs_G1[i,0] += lower_bound_1
            top_idxs_G2[i,0] += lower_bound_1
            top_idxs_G1[i,1] += lower_bound_2
            top_idxs_G2[i,1] += lower_bound_2
            
        gradient_tops1 = np.array((top_idxs_G1[:,1],image_scaled.shape[0]-top_idxs_G1[:,0])).T
        gradient_tops2 = np.array((top_idxs_G2[:,1],image_scaled.shape[0]-top_idxs_G2[:,0])).T

        # Since Grad1 and Grad2 have negative gray values, scale them up before weighting:
        Grad1 = Grad1 + np.max(np.abs(Grad1))
        Grad2 = Grad1 + np.max(np.abs(Grad2))

        # Fascicles
        pos3 = []
        ang3 = []
        weights3 = []
        
        for i in range(len(gradient_tops1)):
            pos3.append(gradient_tops1[i][1])
            ang3.append(theta[gradient_tops1[i][0]])
            weights3.append(Grad1[int(theta[gradient_tops1[i][0]])][gradient_tops1[i][1]-1])
        
        for i in range(len(top_idxs_G2)):
            pos3.append(gradient_tops2[i][1])
            ang3.append(theta[gradient_tops2[i][0]])
            weights3.append(Grad2[int(theta[gradient_tops2[i][0]])][gradient_tops2[i][1]-1])

        # Filter outliers from fascicles
        
        mean_angle3 = np.mean(ang3)
        to_delete = []
        for i in range(len(ang3)):
            if np.abs(ang3[i]-90)<10:
                to_delete.append(i)
        
        ang3 = np.delete(ang3,to_delete,0)
        pos3 = np.delete(pos3,to_delete,0)
        weights3 = np.delete(weights3,to_delete,0)
        
        mean_angle3 = np.mean(ang3)
        to_delete = []
        for i in range(len(ang3)):
            if mean_angle3 > 90 and (ang3[i] < 90 or ang3[i] < mean_angle3 - 10 or ang3[i] > mean_angle3 + 30):
                to_delete.append(i)
                
            if mean_angle3 < 90 and (ang3[i] > 90 or ang3[i] > mean_angle3 + 10 or ang3[i] < mean_angle3 - 30):
                to_delete.append(i)
    
        
        ang3 = np.delete(ang3,to_delete,0)
        pos3 = np.delete(pos3,to_delete,0)
        weights3 = np.delete(weights3,to_delete,0)
        
        # Weight deep aponeurosis
        pos1, ang1, weights1 = np.array(pos1),np.array(ang1),np.array(weights1)
        weighted_ang_deep_apo = weight_angle(ang1, weights1)
        if frame_index == 0:
            weighted_pos_deep_apo = int(np.mean(pos1)) #int(weight_angle(pos1, weights1))
        else:
            weighted_pos_deep_apo = int(0.5 * np.mean(pos1) + 0.5* weighted_pos_deep_apo)
        
        # Weight fascicles
        pos3, ang3, weights3 = np.array(pos3),np.array(ang3),np.array(weights3)
        weighted_ang_fas = weight_angle(ang3, weights3)
        
        if frame_index == 0:
            weighted_pos_fas = int(weight_angle(pos3, weights3))
        else:
            weighted_pos_fas = int(0.5 * np.mean(pos3) + 0.5* weighted_pos_fas)
        
        pos_fascicle.append(weighted_pos_fas)

        # Calculate theta_1 and theta_2
        # and in which direction the line points (positive: up, negative: down (viewing from left to right))
        point_deep_apo = -1*-np.sign(90-weighted_ang_deep_apo)
        theta_1 = abs(90-weighted_ang_deep_apo)

        point_fas = -1*-np.sign(90-weighted_ang_fas)
        theta_2 = abs(90-weighted_ang_fas)

        pennation_angle = theta_1 + theta_2
        
        th1.append(theta_1)
        th2.append(theta_2)
        penn.append(pennation_angle)
        
        if save_images == True:
            img_rgb = plot_results(image_scaled, theta_1, theta_2, 
                                sinogram.shape[0]-weighted_pos_fas,      point_fas, 
                                sinogram.shape[0]-weighted_pos_deep_apo, point_deep_apo, 
                                frame_index)
            
            cv.imwrite(path_to_save_results + '/frames/frame-%04d'%frame_index + '.png', img_rgb)

    th1 = np.array(th1)
    th2 = np.array(th2)
    penn = np.array(penn)
    deep = np.array(pos_deep)
    pos_fascicle = np.array(pos_fascicle)

    # Create folder if it does not exist
    if not os.path.exists(path_to_save_results):
        # Create a new directory because it does not exist 
        os.makedirs(path_to_save_results)

    np.savetxt(path_to_save_results + '/pennation_angle.csv', penn, delimiter=',')
    np.savetxt(path_to_save_results + '/theta_1.csv', th1, delimiter=',')
    np.savetxt(path_to_save_results + '/theta_2.csv', th2, delimiter=',')
    np.savetxt(path_to_save_results + '/pos_deep.csv', deep, delimiter=',')
    np.savetxt(path_to_save_results + '/pos_fas.csv', pos_fascicle, delimiter=',')

    return


if __name__ == "__main__":

    image_dataset_path =  '../../tests/data/image_dataset.hdf5'

    extract_features_from_images(image_dataset_path, 
                                 path_to_save_results='../../results', 
                                 save_images=True)
