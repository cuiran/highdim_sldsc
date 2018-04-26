import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense
from keras import optimizers


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
        cv_loss_array = [] # xval loss for each candidate param
        active_ss_array = get_active_ss_array(data) # put active sumstats into array TODO:write this function
        for param in candidate_params_for_scaled_data:
            cv_loss = 0
            cv_num = 1
            for train_ind,val_ind in kf.split(active_ss_array):
                cv_num+=1
                train_obj = d.data(data.X,data.y,data.weights,train_ind)
                lasso_train = Lasso(alpha=param)
                lasso_train.fit(train_obj)
                val_obj = d.data(data.X,data.y,data.weights,val_ind)
                # TODO: evaluate learned coef and intercept from lasso_train on val_obj, add loss to cv_loss. Reference pyscripts/choose_regparam.py
                val_obj = d.data(data.X,data.y,data.weights,val_ind)
                lasso_train.evaluate(val_obj) # TODO: write this method to generate lasso_train.cv_loss, note, this is weighted loss.
                cv_loss += lasso_train.cv_loss
        cv_loss_array.append(cv_loss)
        # TODO: if the mininum loss occurs at the smallest param, create another list of candidate params that are smaller
        return candidate_params[np.argmin(cv_loss_array)]
                

    def fit(self,data):
        return


def get_candidate_params_for_scaled_data(data):
    # assuming the input data is unscaled, but has attributes mean and std
    
    return 
