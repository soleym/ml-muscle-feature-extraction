import numpy as np
import matplotlib.pyplot as plt

from time import asctime


from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from scipy import signal

from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score


def estimate_penn(penns, PC1_scaled, intercept, slope):
    '''
    Estimate the pennation angle based on the first principal component using 
    the scale and intercept of the best line
    '''
    estimated_scaled_penn = (PC1_scaled - intercept)/slope
    penn_from_scaled = (max(penns) - min(penns))*estimated_scaled_penn + min(penns)
    return penn_from_scaled


def compute_scores(penn_true, penn_pred, printout = True):
    '''
    Compute regression scores for the predicted pennation angles 
    and print the scores if printout is true
    '''
    rmse = np.sqrt(np.mean(np.sum((penn_pred - penn_true)**2))/len(penn_true))
    max_err = max_error(penn_true, penn_pred)
    mse = mean_squared_error(penn_true, penn_pred)
    mae = mean_absolute_error(penn_true, penn_pred)
    r2 = r2_score(penn_true, penn_pred)

    if printout == True:
        print('RMSE      : ' + str(rmse))
        print('MSE       : ' + str(mse))
        print('MAE       : ' + str(mae))
        print('max_error : ' + str(max_err))
        print('R2_score  : ' + str(r2))
    
    return rmse, max_err, mse, mae, r2

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

def get_importances(feats, penns, n_estimators=100, verbose = 0):
    '''
    Get the impurity-based feature importances using a random forest algorithm
    '''
    if verbose > 3:
        verbose = 3
    
    if verbose > 0:
        print(asctime())
        print('Extract feature importances with RF algorithm')
    
    forest = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs = -1, verbose = verbose)
    forest.fit(feats, penns)

    if verbose > 0:
        print(asctime())

    return forest.feature_importances_

def get_number_of_features(importances, total_importance = 0.9, verbose = 0):
    '''
    Find how many of the most important features are needed to have the total
    importance specified
    '''
    sum_importances = 0
    num_feats = 0
    while sum_importances + 1e-6 < total_importance:
        num_feats += 1
        sum_importances = np.sum(np.sort(importances)[-1*num_feats:])
        if verbose > 0:
            print('Total importance for ' + str(num_feats) + ' features: ' + str(sum_importances))
            
    return num_feats

# Find slope and intercept that approximates a line 
def pca_estimator(feats, penns, n_estimators = 100, total_importance = 0.9, plot_figures = True, verbose = 0):
    '''
    Estimate pennation angles based on the first principal component
    total_importance is how large the sum of the selected features should be
    Set verbose to True to print status
    '''
    importances = get_importances(feats = feats, penns = penns, n_estimators=n_estimators, verbose = verbose)
    num_feats = get_number_of_features(importances = importances, total_importance = total_importance, verbose = verbose)

    # Keep only most important features
    important_indices = np.argsort(importances)[-1*num_feats:]
    feats_trimmed = feats[:,important_indices]
    feats_trimmed_scaled = minmax_scale(feats_trimmed)

    pca = PCA(n_components=1,random_state=0)
    pca.fit(feats_trimmed_scaled.T)

    if verbose > 0:
        print()
        print('PCA explained variance ratio: ' + str(pca.explained_variance_ratio_))
        print()

    PC1 = pca.components_[0]
    PC1_scaled = minmax_scale(PC1)
    penns_scaled = minmax_scale(penns)

    if plot_figures == True:
        plt.figure()
        plt.scatter(penns_scaled, PC1_scaled)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('Normalized pennation angle')
        plt.ylabel('Normalized PC1')
        plt.plot(np.unique(penns_scaled), np.poly1d(np.polyfit(penns_scaled, PC1_scaled, 1))(np.unique(penns_scaled)),
                color = 'r')
        
        plt.savefig('PC1_penn_' + str(num_feats) + '.png')
        plt.close()

    slope, intercept = np.polyfit(penns_scaled, PC1_scaled, 1)
    return slope, intercept, important_indices



def cross_val_pca(feats, penns, cv=4, n_estimators = 100, total_importance = 0.9, plot_figures = True, verbose = 0, save_path = ''):
    '''
    Cross validate the pca method for given data and split cv (default is 4)
    '''
    b, a = signal.butter(8, 0.04)
    all_rmse, all_max_error, all_mse, all_mae, all_r2 = [],[],[],[],[]
    
    step_size = len(penns)//cv
    for i in range(cv):
        feat_val = feats[i*step_size:(i+1)*step_size]
        penn_val = penns[i*step_size:(i+1)*step_size]
        feat_train = np.delete(feats, np.arange(i*step_size,(i+1)*step_size), axis=0)
        penn_train = np.delete(penns, np.arange(i*step_size,(i+1)*step_size))
        
        if i == 0 or i == cv-1:
            penn_train = signal.filtfilt(b,a, penn_train, padlen=150)
        else:
            # Smooth train before val
            penn_train[np.arange(0,i*step_size)] = signal.filtfilt(b,a, penn_train[np.arange(0,i*step_size)], padlen=150)
            # Smooth train after val
            penn_train[np.arange(i*step_size,len(penn_train))] = signal.filtfilt(b,a, penn_train[np.arange(i*step_size,len(penn_train))], padlen=150)
            
        penn_val = signal.filtfilt(b,a, penn_val, padlen=150)
        
        slope, intercept, important_indices = pca_estimator(feat_train, penn_train,
                                                            n_estimators = n_estimators,
                                                            total_importance = total_importance, 
                                                            plot_figures = plot_figures, 
                                                            verbose = verbose)
        
        # PCA for validation data
        feats_trimmed = feat_val[:,important_indices]
        feats_trimmed_scaled = minmax_scale(feats_trimmed) 

        pca = PCA(n_components=1,random_state=0)
        pca.fit(feats_trimmed_scaled.T)

        if verbose > 0:
            print()
            print('PCA explained variance ratio (validation): ' + str(pca.explained_variance_ratio_))
            print()

        PC1 = pca.components_[0]
        PC1_scaled = minmax_scale(PC1)
        
        # Predict pennation angles and get scores
        penn_predict = estimate_penn(penn_train, PC1_scaled, intercept, slope)
        # Weakness, cannot predict smaller or greater angles than seen in training set...
        
        plt.figure(figsize = (8,8))

        plt.scatter(np.delete(np.arange(len(penns)), np.arange(i*step_size,(i+1)*step_size), axis=0), 
                 penn_train, label = 'Train', s = 2)
        
        plt.plot(np.arange(i*step_size,(i+1)*step_size), penn_val, label = 'Test')
        plt.plot(np.arange(i*step_size,(i+1)*step_size),penn_predict, label = 'Predicted pennation angles')
        
        plt.xlabel('Frame')
        plt.ylabel('Pennation_angle')
        plt.title('cv'+ str(i))
        plt.legend()
        plt.savefig(save_path +'cv_'+ str(i)+'.png')
        
        np.savetxt('pennation_angle' +str(i)+'.csv', penn_predict, delimiter=',')
        
        rmse, max_err, mse, mae, r2 = compute_scores(penn_val, penn_predict) 

        all_rmse.append(rmse)
        all_max_error.append(max_err)
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
    
    return all_rmse, all_max_error, all_mse, all_mae, all_r2
