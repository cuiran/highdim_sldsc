"""
This file stores discarted codes that are not in use but may be helpful to look at later
"""

def high_dim_sldsc(args):
    if args.standardize:
        # match snps between annotation ld, sumstats and weights, store new files in output-folder, return new file names
        ld_fname,ss_fname,w_fname = d.match_snps(args)
        # split data into training and test according to user specified left out region
        # return array of indices for training and test sets store them in files
        ld_train,ss_train,w_train,ld_test,ss_test,w_test = d.split_data(ld_fname,ss_fname,w_fname,args)        # compute true weights for training data, true weights are the variance of noise
        true_train_weights = d.compute_weights(ld_train,ss_train,w_train,args)
        # create new target by computing chisq-1, store new target
        ttrain_fname = d.new_target(ss_train,args)
        # standardize training ld and store the standardized ld, mean and standard deviation
        stdized_ldtr_fname,mldtr_fname,sldtr_fname = d.standardize_ld(ld_train,args)
        # standardize training ss-1, store it in a new file, together with mean and standard deviation
        stdized_ttr_fname,mttr_fname,sttr_fname = d.standardize_target(ttrain_fname,args)
        # scale train weights so that it has variance 1
        scaled_train_weights = d.scale_train_weights(true_train_weights)
        if args.single-method:
            # run regression with one method
            # store results to file
            coef_fname,bias_fname = r.run_reg_one_method(stdized_ldtr_fname,stdized_ttr_fname,scaled_train_weights,args)
            # use the learned coefs and bias to predict summary statistics on the test data
            p.predict_one_method(coef_fname,bias_fname,mldtr_fname,sldtr_fname,mttr_fname,sttr_fname,args)
        elif args.multi-methods:
            # run regression with multiple methods
            # store results to file
            coef_fname,bias_fname = r.run_reg_multi_methods(stdized_ldtr_fname,stdized_ttr_fname,scaled_train_weights,args)
            p.predict_multi_methods(coef_fname,bias_fname,mldtr_fname,sldtr_fname,mttr_fname,sttr_fname,args)


# the old process function
def process(args,train_data):
    # compute and store true weights
    true_weights = compute_true_w(args,train_data)
    true_w_df = pd.DataFrame(data=true_weights,columns=['TRUE_W'])
    true_w_fname = args.output_folder+'true_weights.txt'
    true_w_df.to_csv(true_w_fname,sep='\t',index=False)
    # compute and store chisq-1
    ss_df = pd.read_csv(train_data.y,delim_whitespace=True).iloc[train_data.active_ind,:]
    chisq = ss_df['CHISQ'].tolist()
    chisq_minus1 = [x-1 for x in chisq]
    minus1_df = pd.DataFrame(data=chisq_minus1,columns=['CHISQ-1'])
    minus1_fname = args.output_folder+'chisq_minus1.txt'
    minus1_df.to_csv(minus1_fname,sep='\t',index=False)
    return data(args.ld,minus1_fname,true_w_fname,train_data.active_ind)

    weights_scaled = d.scale_weights(weights) #TODO: write this function
    s_weights_df = pd.DataFrame(data=weights_scaled,columns=['SCALED_W'])
    weights_fname = args.output_folder+'scaled_weights.txt'
    s_weights_df.to_csv(weights_fname,sep='\t',index=False)

#def get_active(ndarray, active_ind):
#    # for ndarray of dim >=1 get rows that are in active_ind range and concatenate them
#    # output one ndarray with the active_ind sliced out from the input ndarray
#    if ndarray.ndim == 1:
#        start,end = active_ind[0]
#        active_array = ndarray[start:end]
#        for i in range(1,len(active_ind)):
#            start,end = active_ind[i]
#            active_array = np.concatenate((active_array,ndarray[start:end]))
#        return active_array
#    else:
#        active_array = ndarray[active_ind[0][0]:active_ind[0][1],:]
#        for i in range(1,len(active_ind)):
#            start,end = active_ind[i]
#            active_array = np.concatenate(active_array,ndarray[start:end,:],axis=0)
#        return active_array

    @property
    def weighted_stdX(self):
        if not self._weighted_stdX:
            d = u.read_h5(self.X)
            if d.ndim == 1:
                r,c = d.shape[0],1
            else:
                r,c = d.shape
            sum_sqwdiff = 0
            for i in self.active_ind:
                start = i[0]
                end = i[1]
                chunck_data = data(self.X,self.y,self.weights,[[start,end]])
                weighted_chunck = u.compute_weighted_chunck(chunck_data,[0,c])
                sum_sqwdiff += np.sum((weighted_chunck - self.weighted_meanX)**2,axis=0)
            self._weighted_stdX = np.sqrt(np.divide(sum_sqwdiff,self.sum_active_weights))
        return self._weighted_stdX

            if d.ndim == 1:
                r,c = d.shape[0],1
            else:
                r,c = d.shape
            sum_active_weighted_rows = 0
            for i in self.active_ind:
                start = i[0]
                end = i[1]
                chunck_data = data(self.X,self.y,self.weights,[[start,end]])
                weighted_chunck = u.compute_weighted_chunck(chunck_data,[0,c])
                sum_active_weighted_rows += np.sum(weighted_chunck,axis=0)

    @property
    def weighted_meanX(self):
        # weighted average of columns of X (\sum w_ix_i)/(\sum w_i)
        if not self._weighted_meanX:
            d = u.read_h5(self.X)
            if d.ndim == 1:
                r,c = d.shape[0],1
            else:
                r,c = d.shape
            sum_active_weighted_rows = 0
            for i in self.active_ind:
                start = i[0]
                end = i[1]
                chunck_data = data(self.X,self.y,self.weights,[[start,end]])
                weighted_chunck = u.compute_weighted_chunck(chunck_data,[0,c])
                sum_active_weighted_rows += np.sum(weighted_chunck,axis=0)
            self._weighted_meanX = np.divide(sum_active_weighted_rows,self.sum_active_weights)
        return self._weighted_meanX


def concat_data(d,active_ind):
    """ 
    d is a h5py file object, d can be sliced
    active_ind is a list that contains the endpoints of indices
    output ndarray that's the result of concatenate all active indices of d
    """
    start,end = active_ind[0]
    if d.ndim == 1:
        X = d[start:end]
    else:
        X = d[start:end,:]
    if len(active_ind)==1:
        return X
    else:
        for i in active_ind[1:]:
            start,end = i
            X = np.concatenate((X,d[start:end,:]),axis=0)
        return X

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


def get_candidate_params_for_scaled_weighted_data(data):
    #TODO the weighting scheme has changed, so this function needs to change
    # assuming the input data is unscaled, but has attributes mean and std
    scaled_weighted_ydotX = compute_scaled_weighted_ydotX(data) #ndarray
    N = data.active_len
    max_param = np.max(np.divide(np.abs(scaled_weighted_ydotX),N))
    # this formula for max_param is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al.
    candidate_params = compute_candidate_params(max_param)
    return candidate_params


def compute_scaledwydotwX_from_weighted(wydotwX,data):
    # yst\dot Xst = (y\dot X - M\mu_y \mu_X) / (\sigma_y \sigma_X)
    M = data.active_len
    wyst_dot_wXst = np.divide(np.subtract(wydotwX,np.multiply(M,np.multiply(data.weighted_meanX,data.weighted_meany)))
                            ,np.multiply(data.weighted_stdX,data.weighted_stdy))
    return wyst_dot_wXst

def preprocess_data(data):
    # this is for small data
    # center X and y with weighted mean and scale X with the L2 norm of X - weighted_mean(X)
    # no scaling for y
    # return the centered scaled active X and centered active y
    active_X = get_active(read_h5(data.X),data.active_ind)
    active_y = read_chisq_from_ss(data.y,data.active_ind)
    new_X,new_y = center_scale_Xy(active_X,active_y,data)
    return new_X,new_y

    @property
    def X_scale(self):
        # L2 norm of X - weighted_meanX
        centered_norm = []
        for strip in self.X_strips:
            X = u.read_h5(self.X)
            X_active_strip = u.get_strip_active_X(X,strip,self.active_ind)
            X_offset_strip = self.weighted_meanX[strip[0]:strip[1]]
            X_centered_strip = X_active_strip - X_offset_strip
            norm_strip = np.linalg.norm(X_centered_strip,axis=0,ord=2)
            centered_norm.append(norm_strip)
        centered_norm = np.array(centered_norm)
        self._X_scale = centered_norm.flatten()
        return self._X_scale

#TODO improve algorithm
def generator(data,n):
    """
    Generates mini-batches of data
    n = mini batch size
    """
    pdb.set_trace()
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
            yield batch_X,batch_y
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

def get_candidate_params(data):
    # new_X = (X-weighted_meanX)/X_scale and new_y = y - weighted_meany
    # (1/N) * max_j|<new_X_j,new_y>| 
    # this formula is from Regularization Paths for Generalized Linear Models via Coordinate Descent by Friedman et. al. 
    new_ydotX = compute_new_ydotX(data) #ndarray
    max_param = np.max(np.divide(np.abs(new_ydotX),data.N))
    candidate_params = compute_candidate_params(max_param)
    return candidate_params
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

def compute_cvlosses(candidate_params,dd,kf,reg_method):
    """
    candidate_params: a list of candidate parameters
    data: a data object to compute losses on, this is the processed data
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
