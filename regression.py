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
import copy
import line_profiler

class regression:
    def __init__(self,fit_intercept=True,lr=0.01,decay=0.,momentum=0.,minibatch_size=1000,epochs=100):
        self.fit_intercept = fit_intercept
        self.normalize = True # always normalize
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def evaluate_weighted(self,original_data):
        # used self.coef and self.intercept to compute predicted sumstats on data
        # formula for chisq is N*annot_ld*self.coef + self.intercept
        # compute the weighted sum of squared loss on data
        #TODO there's something wrong with this function, most likely it's shuffling related, or maybe it's comparing processed y with predicted original y
        d = u.read_h5(original_data.X)
        val_annotld = u.get_active(d,original_data.active_ind) # output one ndarray represent validation matrix
        val_chisq = u.read_chisq_from_ss(original_data.y,original_data.active_ind)[:,None]# ndarray of validation chisq stats
        val_weights = u.get_active_weights(original_data.weights,original_data.active_ind) # ndarray of validation weights
        if val_annotld.ndim == 1:
            val_annotld = val_annotld[np.newaxis].T
            pred_chisq = np.multiply(original_data.N,val_annotld.dot(self.coef))+self.intercept
        else:
            pred_chisq = np.multiply(original_data.N,val_annotld.dot(self.coef))+self.intercept
        weighted_sumsqerror = val_weights.dot((pred_chisq-val_chisq)**2)
        self.wsse = weighted_sumsqerror
        return weighted_sumsqerror

    def evaluate_no_weight(self,processed_data):
        val_X = u.get_active(u.read_h5(processed_data.X),processed_data.active_ind)
        val_y = u.read_chisq_from_ss(processed_data.y,processed_data.active_ind)[:,None]
        if val_X.ndim == 1:
            val_X = val_X[np.newaxis].T
        pred_y = val_X.dot(self.fitted_coef)
        sse = sum((pred_y - val_y)**2)
        self.cv_loss = sse
        return sse
        
         

class Lasso(regression):
    def __init__(self,alpha='CV',CV_folds=3,CV_epochs=5,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.CV_folds=CV_folds
        self.CV_epochs = CV_epochs

    def choose_param(self,processed_data,original_data):
        # call this method if alpha is set to be 'CV'
        kf = KFold(n_splits = self.CV_folds)
        candidate_params = get_candidate_params(processed_data)
        print('candidate parameters',candidate_params)
        cv_losses = compute_cvlosses(candidate_params,processed_data,original_data,kf,'Lasso')
        best_ind = np.argmin(cv_losses)
        best_alpha = candidate_params[best_ind]
        count = 0
        while count < 5 and best_ind ==4: # best is not the last/smallest
                max_param = candidate_params[best_ind]
                candidate_params = np.array([max_param*(2**-i) for i in range(1,6)])
                cv_losses = compute_cvlosses(candidate_params,processed_data,original_data,kf,'Lasso')
                best_ind = np.argmin(cv_losses)
                best_alpha = candidate_params[best_ind]
                count+=1              
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return best_alpha
    
    def fit(self,processed_data,original_data):

        #  weight and scale the data before fitting model
        if self.alpha == 'CV':
            print('performing cross validation to choose regularization parameter')
            self.alpha = self.choose_param(processed_data,original_data)
            print('choosen alpha',self.alpha)
        print('performing regression')
        model = Sequential()
        model.add(Dense(1,input_dim=processed_data.num_features,use_bias=False))
        self.lr = d.est_lr(processed_data)
        print(self.alpha)
        print(self.lr)
        sgd = optimizers.SGD(lr=self.lr,decay=self.decay,momentum=self.momentum)
        model.compile(loss='mse',optimizer=sgd)
        model.fit_generator(generator(processed_data,self.minibatch_size),
                            steps_per_epoch=processed_data.active_len//self.minibatch_size,
                            epochs=self.epochs,verbose=1,
                            callbacks = [shrink.L1_update(model.trainable_weights[:1],
                                lr=self.lr,regularizer=self.alpha)])
        
        self.fitted_coef= model.get_weights()[0]
        # the learned intercept is the last element in the learned_coef array
        true_coef,true_intercept = recover(self.fitted_coef,processed_data)
        self.coef = true_coef
        self.intercept = true_intercept
        return self.coef,self.intercept,self.alpha

class sk_LassoCV(regression):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def fit(self,processed_data):
        X = u.read_h5(processed_data.X)
        y = u.read_chisq_from_ss(processed_data.y,processed_data.active_ind)
        from sklearn.linear_model import LassoCV
        model = LassoCV(fit_intercept=False)
        model.fit(X,y)
        learned_coef = model.coef_[:,None]
        true_coef,true_intercept = recover(learned_coef,processed_data)
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

def recover(learned_coef,processed_data):
    X_scale = processed_data.X_scale.reshape(1,-1).T
    Ntrue_coef = learned_coef*processed_data.y_scale/X_scale
    true_coef = Ntrue_coef/processed_data.N
    true_intercept = processed_data.y_offset - processed_data.X_offset.dot(Ntrue_coef)
    return true_coef,true_intercept

def dummy_gen(data,n):
    while True:
        X = np.ones((30,690))
        y = np.ones((30,))
        yield X,y

def generator(data,n):
    X = u.read_h5(data.X)[:]
    y = np.array(pd.read_csv(data.y,delim_whitespace=True).iloc[:,0])
    while True:
        for batch_ind in batch_gen(data.active_ind,n):
            ends = u.get_endpoints(batch_ind)
            batch_X = u.get_active_X(X,ends)
            batch_y = u.get_active_y(y,ends)
            yield batch_X,batch_y

def batch_gen(intervals, batch_size):
    """Generates batches from indices_gen(intervals) of size batch_size
    """
    batch = []
    for i in indices_gen(intervals):
        batch.append(i)

        if len(batch) >= batch_size:
            yield batch
            batch = []

def indices_gen(intervals):
    """Generates all indices that are within any of the intervals

    Inputs:
        intervals:  assumed to be of the form [[min1, max1], [min2, max2], ...]
                    where min1 < max1 < min2 < max2 < ...

    Outputs:
        a generator which generates all indices within any of the intervals,
        and goes on forever (repeating)
    """
    while True:
        for interval in intervals:
            [start, end] = interval
            for i in range(start, end):
                yield i

def compute_cvlosses(candidate_params,dd,od,kf,reg_method):
    """
    candidate_params: a list of candidate parameters
    dd: a data object to compute losses on, this is the processed data
    od: original data object contains un-processed data
    kf: object created by KFold, contains indices for train and validation sets
    this function outputs an array of losses for the list of candidate_params
    """
    y = np.array(pd.read_csv(dd.y,delim_whitespace=True).iloc[:,0])
    losses = []
    for p in candidate_params:
        cv_loss = 0
        for train_ind,val_ind in kf.split(y):
            ti = u.get_endpoints(train_ind)
            vi = u.get_endpoints(val_ind)
            tr_obj = copy.copy(dd)
            tr_obj.active_ind = ti
            # validation data with respect to the original data
            val_obj = copy.copy(dd) 
            val_obj.active_ind = u.get_endpoints(u.convert_to_original_ind(od.active_ind,u.expand_ind(vi)))
            if reg_method == 'Lasso':
                cv_lasso = regr.Lasso(alpha = p)
                cv_lasso.epochs = cv_lasso.CV_epochs
                cv_lasso.fit(tr_obj,od)
                cv_lasso.evaluate_no_weight(val_obj)
                cv_loss += cv_lasso.cv_loss
        losses.append(cv_loss)
    return losses

def get_candidate_params(dd):
    # dd is processed data object with dd.X (data.y) storing centered, weighted, scaled X (y)
    # max lambda that yield any interesting (nonzero) coef is
    # (1/N) * max_j|<new_X_j,new_y>|
    # this formula is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al.
    ydotX = compute_ydotX(dd)
    max_param = np.max(np.divide(np.abs(ydotX),dd.active_len)) #TODO check this line, is this the right formula and is dd.N the N in the formula?
    candidates = np.array([max_param*(2**-i) for i in range(1,6)])
    return candidates

def compute_ydotX(dd):
    # dd is processed data
    # compute dot product of y and every column of X
    X = u.read_h5(dd.X)
    y = np.array(pd.read_csv(dd.y,delim_whitespace=True).iloc[:,0])
    dot = []
    for strip in dd.X_strips:
        X_strip = u.get_strip_active_X(X,strip,dd.active_ind)
        dot.append(y.dot(X_strip))
    if len(dot)>1:
        dot = np.concatenate(dot,axis=0)
    return dot
