import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense
from keras import optimizers
from keras import regularizers
import data_processing as d
import reg_main as r
import regression as regr
import useful_functions as u
import pdb
import shrink

class regression:
    def __init__(self,fit_intercept=True,lr=0.01,decay=0.,momentum=0.,minibatch_size=30,epochs=10):
        self.fit_intercept = fit_intercept
        self.normalize = True # always normalize
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def evaluate(self,data):
        # used self.coef and self.intercept to compute predicted sumstats on data
        # formula for chisq is N*annot_ld*self.coef + self.intercept
        # compute the weighted sum of squared loss on data
        #TODO there's something wrong with this function
        d = u.read_h5(data.X)
        val_annotld = u.get_active(d,data.active_ind) # output one ndarray represent validation matrix
        val_chisq = u.read_chisq_from_ss(data.y,data.active_ind)# ndarray of validation chisq stats
        val_weights = u.get_active_weights(data.weights,data.active_ind) # ndarray of validation weights
        if val_annotld.ndim == 1:
            val_annotld = val_annotld[np.newaxis].T
            pred_chisq = np.multiply(data.N,val_annotld.dot(self.coef))+self.intercept
        else:
            pred_chisq = np.multiply(data.N,val_annotld.dot(self.coef))+self.intercept
        weighted_sumsqerror = val_weights.dot((pred_chisq-val_chisq)**2)
        self.cv_loss = weighted_sumsqerror
        return weighted_sumsqerror
        
         

class Lasso(regression):
    def __init__(self,alpha='CV',CV_folds=3,CV_epochs=10,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.CV_folds=CV_folds
        self.CV_epochs = CV_epochs

    def choose_param(self,data):
        # call this method if alpha is set to be 'CV'
        kf = KFold(n_splits = self.CV_folds)
        candidate_params = get_candidate_params(data) 
        cv_losses = compute_cvlosses(candidate_params,data,kf,'Lasso')
        best_alpha = candidate_params[np.argmin(cv_losses)]
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return best_alpha

    def fit(self,data):
        print(self.alpha)
        #  weight and scale the data before fitting model
        if self.alpha == 'CV':
            self.alpha = self.choose_param(data)
            print('choosen alpha',self.alpha)
        model = Sequential()
        model.add(Dense(1,input_dim=data.num_features,use_bias=False))
        sgd = optimizers.SGD(lr=self.lr,decay=self.decay,momentum=self.momentum)
        model.compile(loss='mse',optimizer=sgd)
        model.fit_generator(generator(data,self.minibatch_size),
                            steps_per_epoch=data.active_len//self.minibatch_size,
                            epochs=self.epochs,verbose=1,
                            callbacks = [shrink.L1_update(model.trainable_weights[:1],
                                lr=self.lr,regularizer=self.alpha)])
        learned_coef= model.get_weights()[0]
        # the learned intercept is the last element in the learned_coef array
        true_coef,true_intercept = recover_coef_intercept(data,learned_coef)
        self.coef = true_coef
        self.intercept = true_intercept
        return self.coef,self.intercept,self.alpha

class sk_LassoCV(regression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,data):
        new_X,new_y,X_scale,y_scale = u.preprocess_data(data)
        from sklearn.linear_model import LassoCV
        model = LassoCV(fit_intercept=False)
        model.fit(new_X,new_y)
        learned_coef = model.coef_
        true_coef,true_intercept = recover(data,learned_coef,X_scale,y_scale)
        self.coef = true_coef
        self.intercept = true_intercept
        self.alpha = model.alpha_
        return self.coef,self.intercept,self.alpha
        
class sk_OLS(regression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,data):
        new_X,new_y,X_scale,y_scale = u.preprocess_data(data)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=False)
        model.fit(new_X,new_y)
        learned_coef = model.coef_
        true_coef,true_intercept = recover(data,learned_coef,X_scale,y_scale)
        self.coef = true_coef
        self.intercept = true_intercept
        return self.coef,self.intercept

    def direct_fit(self,data):
        # direct fit skips the standardizing step
        X = u.read_h5(data.X)
        if X.ndim == 1:
            X = X[:].reshape(-1,1)
        active_X = u.get_active(X,data.active_ind)
        active_y = u.read_chisq_from_ss(data.y,data.active_ind)
        w = u.get_active_weights(data.weights,data.active_ind)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(active_X,active_y,sample_weight = w)
        self.coef = model.coef_/data.N
        self.intercept = model.intercept_
        return self.coef, self.intercept

def recover(data,learned_coef,X_scale,y_scale):
    Ntrue_coef = learned_coef*y_scale/X_scale
    true_coef = Ntrue_coef/data.N
    true_intercept = data.mean_y - data.mean_X.dot(Ntrue_coef)
    return true_coef,true_intercept

def recover_coef_intercept(data,learned_coef):
    """ Recover true coef and intercept from the learned coef
    Learned coef is the coefficients for weighted scaled data
    The last element of learned coef is the intercept term
    """
    Ntrue_coef = learned_coef/data.X_scale
    true_coef = Ntrue_coef/data.N
    true_intercept = data.weighted_meany - data.weighted_meanX.dot(Ntrue_coef)
    return true_coef,true_intercept

#TODO improve algorithm
def generator(data,n):
    """
    Generates mini-batches of data
    n = mini batch size
    """
    active_ind = u.expand_ind(data.active_ind) 
    num_active_samples = data.active_len
    annotld = u.read_h5(data.X)
    chisq = np.array(pd.read_csv(data.y,delim_whitespace=True)['CHISQ'])
    all_w = np.array(pd.read_csv(data.weights,delim_whitespace=True).iloc[:,-1])
    num_batches = num_active_samples//n
    while True:
        i=1
        while i<=num_batches:
            batch_ind = active_ind[n*(i-1):n*i]
            batch_X,batch_y = get_batch(data,annotld,chisq,all_w,batch_ind)
            i+=1
            yield batch_X,batch_X
        # the last batch concatenates what remains and the head of data
        batch_ind = active_ind[n*(i-1):]+active_ind[:n-len(active_ind)+n*(i-1)]
        batch_ind.sort()
        batch_X,batch_y = get_batch(data,annotld,chisq,all_w,batch_ind)
        yield batch_X,batch_y

#TODO improve algorithm
def get_batch(data,X,y,w,batch_ind):
    """
    inputs:
    data object
    annotld = h5 dataset for annotation ld
    chisq = entire (not only active) chisq statistics as ndarray
    all_w = ndarray of weights that are ready to be multiplied into y and X
    batch_ind = the indices for this batch
    outputs:
    weighted scaled and centered batch annotation ld as ndarray
    weighted centered batch chisq statistics as ndarray
    """
    batch_y = y[batch_ind]
    batch_w = w[batch_ind]
    if X.ndim == 1:
        batch_X = X[batch_ind]
    else:
        batch_X = X[batch_ind,:]
    sc_X,c_y = u.center_scale_Xy(batch_X,batch_y,data)
    sqrt_w = np.sqrt(batch_w)
    w_sc_X,w_c_y = u.weight_Xy(sqrt_w,sc_X,c_y)
    return w_sc_X,w_c_y   

def compute_cvlosses(candidate_params,data,kf,reg_method):
    """
    candidate_params: a list of candidate parameters
    data: a data object to compute losses on
    kf: object created by KFold, contains indices for train and validation sets
    this function outputs an array of losses for the list of candidate_params
    """
    active_ss_array = u.read_chisq_from_ss(data.y,data.active_ind)
    losses = []
    for param in candidate_params:
        cv_loss = 0
        for train_ind,val_ind in kf.split(active_ss_array):
            train_ind = u.convert_to_original_ind(data.active_ind,train_ind)
            val_ind = u.convert_to_original_ind(data.active_ind,val_ind)
            train_active_ind = d.get_endpoints(train_ind)
            val_active_ind = d.get_endpoints(val_ind)
            train_obj = d.data(data.X,data.y,data.weights,train_active_ind)
            val_obj = d.data(data.X,data.y,data.weights,val_active_ind)
            if reg_method == 'Lasso':
                cv_lasso = regr.Lasso(alpha = param)
                cv_lasso.alpha = param
                cv_lasso.epochs = cv_lasso.CV_epochs
                cv_lasso.fit(train_obj)
                #TODO this is hard coded for Lasso because there's a weird bug about invalid index to scalar variable if I use r.perform_regression
                cv_lasso.evaluate(val_obj) 
                cv_loss += cv_lasso.cv_loss
        losses.append(cv_loss)
    return losses


def get_candidate_params(data):
    # new_X = (X-weighted_meanX)/X_scale and new_y = y - weighted_meany
    # (1/N) * max_j|<new_X_j,new_y>| 
    # this formula is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al. 
    new_ydotX = compute_new_ydotX(data) #ndarray
    max_param = np.max(np.divide(np.abs(new_ydotX),data.N)) 
    candidate_params = compute_candidate_params(max_param)
    return candidate_params

def compute_candidate_params(max_param):
    # given max parameter, create list of candidate params by dividing max_param by 2 for 10 times
    candidate_params = np.array([max_param*(2**-i) for i in range(1,11)])
    return candidate_params

#TODO improve algorithm
def compute_new_ydotX(data):
    # restricted to active_ind
    # compute new_X = (X-weighted_meanX)/X_scale and new_y = y - weighted_meany
    # then compute the dot product of new_y and every column of new_X
    # but realistically we have computational constraints so we need to strip the data into smaller column sizes
    active_w = u.get_active_weights(data.weights,data.active_ind)
    active_y = u.read_chisq_from_ss(data.y,data.active_ind)
    c_y = active_y - data.weighted_meany
    new_y = active_w*active_y
    dot = []
    X = u.read_h5(data.X)
    for strip in data.X_strips:
        start,end = strip
        if X.ndim == 1:
            X = X[:].reshape(-1,1)
        X_strip = u.get_active(X[:,start:end],data.active_ind)
        sc_X = (X_strip - data.weighted_meanX[start:end])/data.X_scale[start:end]
        new_X = active_w.values.reshape(-1,1)*sc_X
        dot.append(new_y.dot(new_X))
    if len(dot) == 1:
        new_dot = dot
    else:
        new_dot = np.concatenate(dot,axis=0)
    return new_dot
