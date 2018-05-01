import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense
from keras import optimizers
import reg_main as r
import useful_functions as u

class regression:
    def __init__(self,fit_intercept=True,lr=0.01,decay=0.,momentum=0.):
        self.fit_intercept = fit_intercept
        self.normalize = True # always normalize
        self.lr = lr
        self.decay = decay
        self.momentum = momentum


class Lasso(regression):
    def __init__(self,alpha='CV',CV_folds=3,epochs=10,**kwarg):
        super().__init__(**kwarg)
        self.alpha=alpha
        self.CV_folds=CV_folds
        self.epochs = epochs

    def choose_param(self,data):
        # call this method if alpha is set to be 'CV'
        kf = KFold(n_splits = self.CV_folds)
        candidate_params_for_scaled_weighted_data = get_candidate_params_for_scaled_weighted_data(data) 
        cv_losses = compute_cvlosses(candidate_params_for_scaled_data,data,kf,'Lasso')
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return candidate_params[np.argmin(cv_losses)] 

    def fit(self,data):
        # TODO finish writing this function
        # first weight the data, then scale the weighted data before fitting model
        return


def compute_cvlosses(candidate_params,data,kf,reg_method):
   """
     candidate_params: a list of candidate parameters
     data: a data object to compute losses on
     kf: object created by KFold, contains indices for train and validation sets
     this function outputs an array of losses for the list of candidate_params
    """
    active_ss_array = u.read_chisq_from_ss(data.y,data.active_ind)
    losses = []
    for param in candidate_params_for_scaled_data:
        cv_loss = 0
        for train_ind,val_ind in kf.split(active_ss_array):
            train_obj = d.data(data.X,data.y,data.weights,train_ind)
            val_obj = d.data(data.X,data.y,data.weights,val_ind)
            reg = r.perform_regression(reg_method,train_obj)
            reg.evaluate(val_obj) #TODO: write evaluate method in regression object, evaluate weighted loss on val_obj
            cv_loss += reg.cv_loss
        losses.append(cv_loss)
    return losses


def get_candidate_params_for_scaled_weighted_data(data):
    # assuming the input data is unscaled, but has attributes mean and std
    scaled_weighted_ydotX = compute_scaled_weighted_ydotX(data) #ndarray
    N = len(data.active_ind)
    max_param = np.max(np.divide(np.abs(scaled_weighted_ydotX),N)) 
    # this formula for max_param is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al.
    candidate_params = compute_candidate_params(max_param)
    return candidate_params

def compute_candidate_params(max_param):
    # given max parameter, create list of candidate params by dividing max_param by 2 for 10 times
    candidate_params = np.array([max_param*(2**-i) for i in range(1,11)])
    return candidate_params

def compute_scaled_weighted_ydotX(data):
    # the idea is to multiply y and X by weights, standardize them then compute the dot product
    # but realistically we have computational constraints so we need to chunck the data
    wy = u.compute_wy(data) #ndarray
    chuncks = u.make_chuncks(data) # indices of column chuncks
    dot = []
    for ind in chuncks:
        wchunck = compute_weighted_chunck(data,ind)
        dot.append(wy.dot(wchunck))
    wydotwX = np.concatenate(dot,axis=0)
    scaled_weighted_ydotX = compute_scaledwydotwX_from_weighted(wydotwX,data)
    return scaled_weighted_ydotX


def compute_scaledwydotwX_from_weighted(wydotwX,data):
    # yst\dot Xst = (y\dot X - M\mu_y \mu_X) / (\sigma_y \sigma_X)
    M = len(data.active_ind)
    wyst_dot_wXst = np.divide(np.subtract(wydotwX,np.multiply(M,np.multiply(data.weighted_meanX,data.weighted_meany))),np.multiply(data.weighted_stdX,weighted_stdy))
    return wyst_dot_wXst
