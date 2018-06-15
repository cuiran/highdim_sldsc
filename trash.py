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

