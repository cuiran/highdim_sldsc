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

class regression:
    def __init__(self,fit_intercept=True,lr=0.01,decay=0.,momentum=0.,minibatch_size=30,epochs=20):
        self.fit_intercept = fit_intercept
        self.normalize = True # always normalize
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def evaluate(self,original_data):
        # used self.coef and self.intercept to compute predicted sumstats on data
        # formula for chisq is N*annot_ld*self.coef + self.intercept
        # compute the weighted sum of squared loss on data
        #TODO there's something wrong with this function
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
        self.cv_loss = weighted_sumsqerror
        return weighted_sumsqerror
        
         

class Lasso(regression):
    def __init__(self,alpha='CV',CV_folds=3,CV_epochs=10,**kwargs):
        super().__init__(**kwargs)
        self.alpha=alpha
        self.CV_folds=CV_folds
        self.CV_epochs = CV_epochs

    def choose_param(self,processed_data,original_data):
        # call this method if alpha is set to be 'CV'
        kf = KFold(n_splits = self.CV_folds)
        candidate_params = get_candidate_params(processed_data) 
        cv_losses = compute_cvlosses(candidate_params,processed_data,original_data,kf,'Lasso')
        best_alpha = candidate_params[np.argmin(cv_losses)]
        # TODO: if the mininum loss occurs at the smallest param, try smaller candidate params
        return best_alpha

    def fit(self,processed_data,original_data):
        #  weight and scale the data before fitting model
        if self.alpha == 'CV':
            self.alpha = self.choose_param(processed_data,original_data)
            print('choosen alpha',self.alpha)
        model = Sequential()
        model.add(Dense(1,input_dim=processed_data.num_features,use_bias=False))
        sgd = optimizers.SGD(lr=self.lr,decay=self.decay,momentum=self.momentum)
        model.compile(loss='mse',optimizer=sgd)
        model.fit_generator(generator(processed_data,self.minibatch_size),
                            steps_per_epoch=processed_data.active_len//self.minibatch_size,
                            epochs=self.epochs,verbose=1,
                            callbacks = [shrink.L1_update(model.trainable_weights[:1],
                                lr=self.lr,regularizer=self.alpha)])
        learned_coef= model.get_weights()[0]
        # the learned intercept is the last element in the learned_coef array
        true_coef,true_intercept = recover(learned_coef,processed_data)
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

def recover(learned_coef,processed_data):
    X_scale = processed_data.X_scale.reshape(1,-1).T
    Ntrue_coef = learned_coef*processed_data.y_scale/X_scale
    true_coef = Ntrue_coef/processed_data.N
    true_intercept = processed_data.y_offset - processed_data.X_offset.dot(Ntrue_coef)
    return true_coef,true_intercept

#def generator(data,n):
#    """
#    Generates mini-bathces of data
#    data object should be processed already. Processed data.y only has one column
#    n = mini batch size
#    """
#    pdb.set_trace()
#    X = u.read_h5(data.X)
#    y = np.array(pd.read_csv(data.y,delim_whitespace=True).iloc[:,0])
#    b = data.active_len//n # number of complete batches from one iteration
#    while True:
#        i=1
#        while i <= b:
#            batch_X = X[n*(i-1):n*i,:]
#            batch_y = y[n*(i-1):n*i]
#            i+=1
#            yield batch_X,batch_y
#        # the last batch consist of what's remaining in the data and the head of data
#        remaining_X = X[n*(i-1):,:]
#        need_from_head = n - remaining_X.shape[0]
#        batch_X = np.concatenate((remaining_X,X[:need_from_head,:]),axis=0)
#        remaining_y = y[n*(i-1):]
#        batch_y = np.concatenate((remaining_y,y[:need_from_head]),axis=0)
#        yield batch_X,batch_y

def generator(data,n):
    X = u.read_h5(data.X)
    y = np.array(pd.read_csv(data.y,delim_whitespace=True).iloc[:,0])
    while True:
        pdb.set_trace()
        j = 1 # which interval of active_ind we are on
        while j <= len(data.active_ind):
            #TODO this doesn't work when active_ind has multiple intervals
            i = 1 # batch number
            start = data.active_ind[j-1][0]
            end = data.active_ind[j-1][1]
            b = (end - start)//n
            while i<=b:
                batch_X = X[n*(i-1)+start:n*i+start,:]
                batch_y = y[n*(i-1)+start:n*i+start]
                i+=1
                yield batch_X,batch_y
            pdb.set_trace()
            j += 1
             
            remaining_X = X[n*(i-1)+start:end,:]
            need_from_next = n - remaining_X.shape[0]
            batch_X = np.concatenate((remaining_X,X[start:need_from_head+start,:]),axis=0)
            remaining_y = y[n*(i-1)+start:end]
            batch_y = np.concatenate((remaining_y,y[start:need_from_head+start]),axis=0)
            j+=1
            yield batch_X,batch_y

def generator(data,n):
    X = u.read_h5(data.X)
    y = np.array(pd.read_csv(data.y,delim_whitespace=True).iloc[:,0])
    while True:
        

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
            val_obj = copy.copy(od)
            val_obj.active_ind = u.get_endpoints(u.convert_to_original_ind(od.active_ind,u.expand_ind(vi)))
            if reg_method == 'Lasso':
                cv_lasso = regr.Lasso(alpha = p)
                cv_lasso.epochs = cv_lasso.CV_epochs
                cv_lasso.fit(tr_obj,od)
                cv_lasso.evaluate(val_obj)
                cv_loss += cv_lasso.cv_loss
        losses.append(cv_loss)
    return losses

def get_candidate_params(dd):
    # dd is processed data object with dd.X (data.y) storing centered, weighted, scaled X (y)
    # max lambda that yield any interesting (nonzero) coef is
    # (1/N) * max_j|<new_X_j,new_y>|
    # this formula is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al.
    ydotX = compute_ydotX(dd)
    max_param = np.max(np.divide(np.abs(ydotX),dd.N))
    candidates = np.array([max_param*(2**-i) for i in range(1,11)])
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
