import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense
from keras import optimizers
import reg_main as r


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
        candidate_params_for_scaled_data = get_candidate_params_for_scaled_data(data) 
        cv_losses = compute_cvlosses(candidate_params_for_scaled_data,data,kf,'Lasso')
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return candidate_params[np.argmin(cv_losses)] 

    def fit(self,data):
        return


def compute_cvlosses(candidate_params,data,kf,reg_method):
   """
     candidate_params: a list of candidate parameters
     data: a data object to compute losses on
     kf: object created by KFold, contains indices for train and validation sets
     this function outputs an array of losses for the list of candidate_params
    """
    active_ss_array = get_active_ss_array(data) # get active sumstats array TODO: write this function
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


def get_candidate_params_for_scaled_data(data):
    # assuming the input data is unscaled, but has attributes mean and std
    # TODO finish writing this function, refer to pyscripts/choose_regparam.py
        
    return 
