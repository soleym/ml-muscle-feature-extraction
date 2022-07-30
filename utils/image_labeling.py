import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def weight_angle(angles, weights):
    '''
    Parameters
    ----------
    angles : numpy array of angles (or positions)
        The elements are angles corresponding to vast peak points 
    weights : numpy array
        The elements are weights, i.e. gray values of vast peak points 
        (used to weigh orientation and position of straight line)
    Returns
    -------
    The weighted angle

    '''
    return np.sum(angles*weights)/np.sum(weights)
    
def measure_line(theta, pos, point, image):
    '''
    Parameters
    ----------
    theta : theta_1 or theta_2, angle with the horizontal line 
    pos   : The weighted position of the feature
    point : The direction that the line points into (+1 or -1) 
            (positive: up, negative: down (viewing from left to right))
    image : The image to plot the lines on
    Returns
    -------
    line_start: The left endpoint of the line
    line_stop:  The rigth endpoint of the line
    '''
    x,y = image.shape[1], image.shape[0]
    line_start = (0, int(pos-point*0.5*x*np.tan(np.radians(theta))))
    line_stop = (x-1,int(pos+point*0.5*x*np.tan(np.radians(theta))))
    
    return line_start, line_stop
    
def get_reference_max(img_data, factor):
    '''
    Get the reference max for the image data for a given factor
    '''
    max_value = np.abs(img_data).max()
    reference_max = factor * max_value
    return reference_max

def get_peaks(grad, num_peaks):
    '''
    Get the indices of the peak points of a (gradient) radon transform 
    for a given numper of peaks
    '''
    grad_copy = np.copy(grad)
    top_idxs_grad = []
    for i in range(num_peaks):
        index = np.unravel_index(grad_copy.argmax(), grad_copy.shape)
        grad_copy[index] = np.min(grad)-1
        index = np.array(index)
        top_idxs_grad.append(index)
        
    return np.array(top_idxs_grad)

def pre_cluster_outlier_removal(tops, theta, res_pos):
    '''
    Filter peak points from corners of the gradient radon transform
    prior to clustering. Inputs are the peak indices, theta and the position resolution
    '''
    to_delete = []
    for i in range(len(tops)):
        if (theta[tops[i][0]-1] < 180*0.25 and tops[i][1] - 1 < res_pos*0.25):
            to_delete.append(i)
        if (theta[tops[i][0]-1] > 180*0.75 and tops[i][1] - 1 > res_pos*0.75): 
            to_delete.append(i)
        if (theta[tops[i][0]-1] < 180*0.25 and tops[i][1] - 1 > res_pos*0.75):
            to_delete.append(i)
        if (theta[tops[i][0]-1] > 180*0.75 and tops[i][1] - 1 < res_pos*0.25):
            to_delete.append(i)
        if tops[i][1] - 1 > res_pos*0.95:
            to_delete.append(i)
    return np.delete(tops,to_delete,0)

def order_labels(tops, pred):
    '''
    Reorder labels for a prediction vector such that sup-aps have label 0 (red), 
    fascicles have label 1 (green) and deep-aps have label 2 (blue)
    '''
    label10 = pred[np.argsort(tops[:,1])][0]
    label12 = pred[np.argsort(tops[:,1])][-1]

    for i in range(len(tops[:,1])):
        if pred[i] == label10:
            pred[i] = 10
        elif pred[i] == label12:
            pred[i] = 12
        else:
            pred[i] = 11
    
    return pred - 10


def load_rgb(image):
    '''
    Turn an array into a grayscale image with three channels, so colored
    lines and text can be drawn on the image
    '''
    plt.imsave('temp.png',image)
    plt.close()
    img_rgb = cv.imread('temp.png')
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Convert grayscale to rgb/bgr by duplicating the channels
    # This is necessary to plot colored lines on top of the image
    img_gray = []
    for i in range(len(img_rgb)):
        row = []
        for j in range(len(img_rgb[i])):
            row.append([img_rgb[i][j], img_rgb[i][j], img_rgb[i][j]])
        img_gray.append(row)

    return np.array(img_gray)


def draw_text_on_image(image, text, position, color = (0,0,255)):
    '''
    Draw text on an image, given the position (bottom left corner of text)
    '''
    cv.putText(image,
               text, 
               position,                  # bottom left corner of text
               cv.FONT_HERSHEY_SIMPLEX,  # font
               0.5,                       # font scale
               color,                     # font color
               1)                         # line type


def plot_results(image, theta_1, theta_2, weighted_pos_fas, point_fas, weighted_pos_deep_apo, point_deep_apo, frame_index):
    '''
    Draw the results given positions, orientations and angles of the deep aponeurosis and fascicle
    Returns an image which can then be saved
    '''
    img_rgb = load_rgb(image)

    # Draw the weighted fascicles and deep aponeuroses on the image
    line_start_fas, line_stop_fas   = measure_line(theta_2, weighted_pos_fas, point_fas, img_rgb)
    line_start_deep, line_stop_deep = measure_line(theta_1, weighted_pos_deep_apo, point_deep_apo, img_rgb)

    cv.line(img_rgb,line_start_fas,line_stop_fas,(0,255,0),2) # green
    cv.line(img_rgb,line_start_deep,line_stop_deep,(255,0,0),2) # blue

    draw_text_on_image(img_rgb, 'Pennation angle: %.2f' % (theta_1 + theta_2), (10,50))
    draw_text_on_image(img_rgb, 'Frame index: ' + str(frame_index),        (10,20))
    
    return img_rgb

def plot_double_results(image, theta_1, theta_2, penn_ANN, weighted_pos_fas, point_fas, weighted_pos_deep_apo, point_deep_apo, frame_index):
    '''
    Draw the results given positions, orientations and angles of the deep aponeurosis and fascicle from ALT
    and pennation angle from ANN
    Returns an image which can then be saved
    '''
    img_rgb = load_rgb(image)

    # Draw the weighted fascicles and deep aponeuroses on the image
    line_start_fas, line_stop_fas   = measure_line(theta_2, weighted_pos_fas, point_fas, img_rgb)
    line_start_deep, line_stop_deep = measure_line(theta_1, weighted_pos_deep_apo, point_deep_apo, img_rgb)

    line_start_fas_ann, line_stop_fas_ann   = measure_line(penn_ANN - theta_1, weighted_pos_fas, point_fas, img_rgb)

    cv.line(img_rgb,line_start_fas,line_stop_fas,(0,255,0),2) # green
    cv.line(img_rgb,line_start_deep,line_stop_deep,(255,0,0),2) # blue
    
    cv.line(img_rgb,line_start_fas_ann,line_stop_fas_ann,(0,0,255),2) #  red

    cv.rectangle(img_rgb, (5,5), (220,100), (0,0,0), -1)
    draw_text_on_image(img_rgb, 'Penn. angle (ALT): %.2f' % (theta_1 + theta_2), (10,50), (0,255,0)) 
    draw_text_on_image(img_rgb, 'Penn. angle (ML): %.2f' % penn_ANN, (10,80), (0,0,255)) 
    draw_text_on_image(img_rgb, 'Frame index: ' + str(frame_index), (10,20), (200,200,200))
    
    return img_rgb


def filter_canny_aponeuroses(ang, pos, weights):
    '''
    Filter aponeuroses that 
    1. Are greater than 15 degrees
    2. Point in the opposite direction than the majority
    3. Are at the edge of the image
    4. Are positioned very far from the average position
    Used where aponeuroses are detected with canny edge detection
    '''

    to_delete = []  

    for i in range(len(pos)):
        if np.abs(ang[i] - 90) > 15:
            to_delete.append(i) 
            
    ang = np.delete(ang,to_delete,0)
    pos = np.delete(pos,to_delete,0)
    weights = np.delete(weights,to_delete,0)
    
    to_delete = []  
    vote1 = 0
    vote2 = 0
    for i in range(len(ang)):
        if ang[i] > 90:
            vote1+=1
        if ang[i] < 90: # No vote if exactly 90
            vote2 += 1
    
    if vote1>vote2:
        for i in range(len(ang)):
            if ang[i] < 90:
                to_delete.append(i)
    else:
        for i in range(len(ang)):
            if ang[i]> 90:
                to_delete.append(i)
            
    ang = np.delete(ang,to_delete,0)
    pos = np.delete(pos,to_delete,0)
    weights = np.delete(weights,to_delete,0)
    
    '''
    print(ang)
    print(pos)
    to_delete = []  
    for i in range(len(pos)):
        if pos[i] < 5:
            to_delete.append(i) 
            
    ang = np.delete(ang,to_delete,0)
    pos = np.delete(pos,to_delete,0)
    weights = np.delete(weights,to_delete,0)
    
    print(ang)
    print(pos)
    to_delete = []  
    mean_pos = np.mean(pos)
 
    for i in range(len(pos)):
        if np.abs(pos[i] - mean_pos) > 40:
            to_delete.append(i) 
            
    ang = np.delete(ang,to_delete,0)
    pos = np.delete(pos,to_delete,0)
    weights = np.delete(weights,to_delete,0)
    print(ang)
    print(pos)
    
    '''
    return ang, pos, weights





    
