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
    def __init__(self,fit_intercept=True,lr=0.01,decay=0.,momentum=0.,minibatch_size=30,epochs=200):
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
        d = u.read_h5(data.X)
        val_annotld = u.get_active(d,data.active_ind) # output one ndarray represent validation matrix
        val_chisq = u.read_chisq_from_ss(data.y,data.active_ind) # ndarray of validation chisq stats
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
    def __init__(self,alpha='CV',CV_folds=3,CV_epochs=100,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.CV_folds=CV_folds
        self.CV_epochs = CV_epochs

    def choose_param(self,data):
        # call this method if alpha is set to be 'CV'
        kf = KFold(n_splits = self.CV_folds)
        candidate_params_for_scaled_weighted_data = get_candidate_params_for_scaled_weighted_data(data) 
        cv_losses = compute_cvlosses(candidate_params_for_scaled_weighted_data,data,kf,'Lasso')
        print(candidate_params_for_scaled_weighted_data[np.argmin(cv_losses)])
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return candidate_params_for_scaled_weighted_data[np.argmin(cv_losses)] 

    def fit(self,data):
        print(self.alpha)
        #  weight and scale the data before fitting model
        if self.alpha == 'CV':
            self.alpha = self.choose_param(data)
            print('choosen alpha',self.alpha)
        model = Sequential()
        model.add(Dense(1,input_dim=data.num_features+1,use_bias=False))
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
        X,y = u.preprocess_data(data)
        wX,wy = u.weight_Xy(data,X,y)
        from sklearn.linear_model import LassoCV
        model = LassoCV(fit_intercept=False)
        model.fit(wX,wy)
        learned_coef = model.coef_
        true_coef = (learned_coef/data.X_scale)/data.N
        true_intercept = data.weighted_meany - data.weighted_meanX.dot(learned_coef/data.X_scale)
        self.coef = true_coef
        self.intercept = true_intercept
        self.alpha = model.alpha_
        return self.coef,self.intercept,self.alpha
        
    def direct_fit(self,data):
        # direct fit without manually weight and scale the data
        X = u.read_h5(data.X)
        if X.ndim == 1:
            X = X[:].reshape(-1,1)
        active_X = u.get_active(X,data.active_ind)
        active_y = u.read_chisq_from_ss(data.y,data.active_ind)
        w = u.get_active_weights(data.weights,data.active_ind)
        from sklearn.linear_model import LassoCV
        model = LassoCV()
        model.fit(active_X,active_y,sample_weight = w)
        self.coef = model.coef_/data.N
        self.intercept = model.intercept_
        self.alpha = model.alpha_
        return self.coef,self.intercept,self.alpha

class sk_OLS(regression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,data):
        X,y = u.preprocess_data(data)
        wX,wy = u.weight_Xy(data,X,y) 
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=False)
        model.fit(wX,wy)
        learned_coef = model.coef_
        true_coef = (learned_coef/data.X_scale)/data.N
        true_intercept = data.weighted_meany - data.weighted_meanX.dot(learned_coef/data.X_scale)
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

   

def recover_coef_intercept(data,learned_coef):
    """ Recover true coef and intercept from the learned coef
    Learned coef is the coefficients for weighted scaled data
    The last element of learned coef is the intercept term
    """
    mean_active_w,std_active_w = u.get_mean_std_w(data)
    weighted_stdX_with_intercept = np.append(data.weighted_stdX,std_active_w)
    coefs = learned_coef.flatten()
    Ntrue_coef = data.weighted_stdy*(coefs/weighted_stdX_with_intercept)
    true_coef = np.divide(Ntrue_coef[:-1],data.N)
    true_intercept = Ntrue_coef[-1]
    return true_coef,true_intercept

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
    mean_active_w,std_active_w = u.get_mean_std_w(data)
    stdized_w = (all_w - mean_active_w)/std_active_w # ideally we should standardize according to active_w without disrupting the indices
    num_batches = num_active_samples//n
    while True:
        i=1
        while i<=num_batches:
            batch_ind = active_ind[n*(i-1):n*i]
            batch_ws_annotld,batch_ws_chisq = get_batch(data,annotld,chisq,all_w,stdized_w,batch_ind)
            i+=1
            yield batch_ws_annotld,batch_ws_chisq
        # the last batch concatenates what remains and the head of data
        batch_ind = active_ind[n*(i-1):]+active_ind[:n-len(active_ind)+n*(i-1)]
        batch_ind.sort()
        batch_ws_annotld,batch_ws_chisq = get_batch(data,annotld,chisq,all_w,stdized_w,batch_ind)
        yield batch_ws_annotld,batch_ws_chisq

def get_batch(data,annotld,chisq,all_w,stdized_w,batch_ind):
    """
    inputs:
    data object
    annotld = h5 dataset for annotation ld
    chisq = entire (not only active) chisq statistics as ndarray
    all_w = ndarray of weights that are ready to be multiplied into y and X
    stdized_w = scaled full ready_to_multiply weights as ndarray
    batch_ind = the indices for this batch
    outputs:
    weighted scaled batch annotation ld as ndarray
    weighted scaled batch chisq statistics as ndarray
    """
    batch_chisq = chisq[batch_ind]
    if annotld.ndim == 1:
        batch_annotld = annotld[batch_ind]
    else:
        batch_annotld = annotld[batch_ind,:]
    batch_w = all_w[batch_ind]
    batch_stdized_w = stdized_w[batch_ind]
    if batch_annotld.ndim == 1:
        w_annotld = batch_w*batch_annotld
    else:
        w_annotld = np.multiply(batch_w[np.newaxis].T,batch_annotld) # w*X
    w_chisq = np.multiply(batch_w,batch_chisq) # w*y
    pdb.set_trace()
    ws_annotld = (w_annotld - data.weighted_meanX)/data.weighted_stdX
    ws_annotld_with_intercept = u.attach_column(ws_annotld,batch_stdized_w) # add column of stdized_tomulti_w to regressors
    ws_chisq = (w_chisq - data.weighted_meany)/data.weighted_stdy
    return ws_annotld_with_intercept,ws_chisq

   

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


def get_candidate_params_for_scaled_weighted_data(data):
    # assuming the input data is unscaled, but has attributes mean and std
    scaled_weighted_ydotX = compute_scaled_weighted_ydotX(data) #ndarray
    N = data.active_len
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
    chuncks = u.make_strips(data) # indices of column chuncks
    dot = []
    for ind in chuncks:
        wchunck = u.compute_weighted_chunck(data,ind)
        dot.append(wy.dot(wchunck))
    if len(dot) == 1:
        wydotwX = dot
    else:
        wydotwX = np.concatenate(dot,axis=0)
    scaled_weighted_ydotX = compute_scaledwydotwX_from_weighted(wydotwX,data)
    return scaled_weighted_ydotX


def compute_scaledwydotwX_from_weighted(wydotwX,data):
    # yst\dot Xst = (y\dot X - M\mu_y \mu_X) / (\sigma_y \sigma_X)
    M = data.active_len
    wyst_dot_wXst = np.divide(np.subtract(wydotwX,np.multiply(M,np.multiply(data.weighted_meanX,data.weighted_meany)))
                            ,np.multiply(data.weighted_stdX,data.weighted_stdy))
    return wyst_dot_wXst
