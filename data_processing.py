import h5py
import pdb
import useful_functions as u
#from memory_profiler import profile
from os import path
import copy
import pandas as pd
import numpy as np
import line_profiler

class data:
    def __init__(self,X,y,weights,active_ind):
        """
        X is the .h5 filename that contains annotation ld with #SNP rows and #cell types columns, no header
        y is the file name of text file containing formatted summary statistics information 
            columns of such file are in order: SNP, A1, A2, N, CHISQ, Z
        weights is the file name of text file that contain weights related information, 
            it can be weights ld or the final weights
        active_ind is a list that contains the endpoints of indices that are currently active.
            For exampla if indices [1,2,4,5,6,8,9,12] are the active indices then 
            active_ind equals to [[1,3],[4,7],[8,10],[12,13]]
        active_len is the length of active rows
        mean_X is the mean taken over rows of active X
        std_X is the standard deviation taken over rows of active X
        mean_y is the mean taken over active elements in CHISQ column of y
        std_y is the standard deviation taken over active elements in CHISQ column of y
        ...
        N is the number of individuals
        """
        self.X = X 
        self.y = y
        self.weights = weights
        self.active_ind = active_ind
        self._active_len = None
        self._mean_X = None
        self._std_X = None
        self._mean_y = None
        self._std_y = None
        self._N = None
        self._num_features = None
        self._X_strips = None
        self._Xstripsize = None
        self.X_offset = None
        self.X_scale = None

    @property
    def Xstripsize(self):
        if self.num_features < 100:
            s = 1
        else:
            s = 100
        return s

    @property
    def num_features(self):
        if not self._num_features:
            d = u.read_h5(self.X)
            if d.ndim == 1:
                self._num_features = 1
            else:
                self._num_features = d.shape[1]
        return self._num_features

    @property
    def active_len(self):
        if not self._active_len:
            length = 0
            for i in self.active_ind:
                length += i[1]-i[0]
            self._active_len = length
        return self._active_len
#TODO improve algorithm
    @property
    def mean_X(self):
        if self._mean_X is None:
            X = u.read_h5(self.X)
            mean = []
            std = []
            for strip in self.X_strips:
                X_strip = u.get_strip_active_X(X,strip,self.active_ind)
                mean_strip = np.mean(X_strip,axis=0)
                mean.append(mean_strip)
                std_strip = np.std(X_strip,axis=0)
                std.append(std_strip)
            mean = np.array(mean)
            std = np.array(std)
            self._mean_X = np.concatenate(mean)
            self._std_X = np.concatenate(std)
        return self._mean_X

    @property
    def std_X(self):
        if not self._std_X:
            self.mean_X(self)
        return self._std_X

    @property
    def mean_y(self):
        if not self._mean_y:
            y = u.read_chisq_from_ss(self.y,self.active_ind)
            self._mean_y = np.mean(y)
            self._std_y = np.std(y)
        return self._mean_y

    @property
    def std_y(self):
        if not self._std_y:
            self.mean_y(self)
        return self._std_y
 
    @property
    def X_strips(self):
        # stripping X by column, return a list of lists of endpoints
        self._X_strips = u.make_strips(self,stripsize=self.Xstripsize)
        return self._X_strips

    @property
    def N(self):
        if not self._N:
            ss_df = pd.read_csv(self.y,delim_whitespace=True)
            self._N = np.mean(ss_df['N'])
        return self._N


def match_SNPs(args):
    annot_snps = pd.read_csv(args.annot_snplist,delim_whitespace=True)['SNP'].tolist()
    ss_snps = pd.read_csv(args.sumstats,delim_whitespace=True)['SNP'].tolist()
    if annot_snps==ss_snps:
        active_ind = [x for x in range(len(ss_snps))]
        return data(args.ld,args.sumstats,args.weights_ld,active_ind)
    else:
        raise ValueError('--ld and --sumstats must have the same SNPs')


def get_traintest_ind(args):
    chr_list = ['chr'+str(i) for i in range(1,23)]
    if args.leave_out in chr_list:
        chrsnp = pd.read_csv(args.annot_snplist,delim_whitespace=True)
        chr_num = int(args.leave_out[3:])
        test_ind = chrsnp.index[chrsnp['CHR']==chr_num].tolist()
        all_ind = range(chrsnp.shape[0])
        train_ind = [x for x in all_ind if x not in test_ind]
    else:
        #TODO:implement ability to process of a file of specified SNPs to leave out
        print('--leave-out functionality is not complete.')
    final_train_ind = u.get_endpoints(train_ind)
    final_test_ind = u.get_endpoints(test_ind)
    return final_train_ind,final_test_ind

def concat_weights(weights_fname,chr_list):
    # for the chromosomes in chr_list, concatenates corresponding weights
    # weights_fname is the file name that stops right before chrom number
    # for example if weights files are "weights.1.l2.ldscore.gz" ... "weights.22.l2.ldscore.gz"
    # then weights_fname should be "weights."
    # chr_list is the list of strings ['1','2'...] representing chromosomes that we would like to concatenate the weights of
    # assuming the column contains weights information is the last column
    weights_fnames = [weights_fname + x + '.l2.ldscore.gz' for x in chr_list]
    weights_dfs = [pd.read_csv(x,delim_whitespace=True) for x in weights_fnames]
    to_concat = [df.iloc[:,-1] for df in weights_dfs]
    weights = np.concatenate(to_concat,axis=0)
    return weights

def compute_final_w(args,data):
    # concatenate weights in args.weights_ld
    # this is the part of final weights that correct some of the correlated errors
    chr_list = [str(i) for i in range(1,23)]
    concated_weights = concat_weights(data.weights,chr_list) 
    weights_corr = np.fmax(concated_weights,1.0) # prevent inverse from being too large
    # compute weights that correct some of the heteroskedasticity
    M = float(data.active_len)
    sum_ld = u.h5_sum_all(data.X,data.active_ind)
    sum_ss = u.chisq_sum_all(data.y,data.active_ind)
    l = sum_ld/M
    s = sum_ss/M
    Ntau_hat = np.divide(s-1,l)
    weights_hetero = 2*((Ntau_hat*u.h5_sum_cols(data.X,data.active_ind)+1)**2)
    # multiply heteroskedasticity weights with correlation weights
    final_weights = np.multiply(weights_hetero,weights_corr)
    final_weights = np.fmax(final_weights,1.0)
    return final_weights

def weights_processing(args,original_data):
    print('processing weights')
    final_weights = compute_final_w(args,original_data)
    weights_inv = 1/final_weights
    df = pd.DataFrame(data=weights_inv,columns=['WEIGHT'])
    print('saving weights to '+ args.output_folder)
    weights_fname = args.output_folder+'final_weights.txt'
    df.to_csv(weights_fname,sep='\t',index=False)
    return weights_fname

def init_new_data(old_data):
    new_data = copy.copy(old_data)
    new_data.X = old_data.X+'_processed.h5'
    new_data.y = old_data.y+'_processed.txt'
    new_data.weights = old_data.weights+'_processed.txt'
    new_data.active_ind = [[0,old_data.active_len]]
    new_data.X_offset = old_data.mean_X
    new_data.X_scale = []
    return new_data

def preprocess_large_no_wsh(dd):
    print('processing data no weight no shuffling')
    # first argument is a data object with original X, y and weights info
    # center scale X and y (no weighting or shuffling) 
    X = u.read_h5(dd.X)
    new_data = init_new_data(dd)

def preprocess_large(dd,snplist):
    print('processing data') 
    # data object with original X, y and weights info
    # center, weight, scale X, and y and store the result in a new data object
    # store X_offset, y_offset, X_scale, and y_scale too
    X = u.read_h5(dd.X)
    w = u.get_active_weights(dd.weights,dd.active_ind)
    sqrt_w = np.sqrt(np.array(w))
    new_data = init_new_data(dd)
    # center weight scale X
    first_strip = dd.X_strips[0]
    X_active_strip = u.get_strip_active_X(X,first_strip,dd.active_ind)
    if X_active_strip.ndim == 1:
        X_active_strip = X_active_strip[:,None]
    print('centering weighting scaling X')
    centered_strip = X_active_strip - new_data.X_offset[first_strip[0]:first_strip[1]]
    weighted_strip = sqrt_w[:,None]*centered_strip
    X_strip_scale = np.std(weighted_strip,axis=0)
    new_data.X_scale.append(X_strip_scale)
    new_X_strip = weighted_strip/X_strip_scale
    with h5py.File(new_data.X,'w') as f:
        f.create_dataset('dataset',maxshape=(dd.active_len,X.shape[1]),data=new_X_strip)
    for strip in dd.X_strips[1:]:
        X_active_strip = u.get_strip_active_X(X,strip,dd.active_ind)
        centered_strip = X_active_strip - dd.mean_X[strip[0]:strip[1]]
        weighted_strip = sqrt_w[:,None]*centered_strip
        X_strip_scale = np.std(weighted_strip,axis=0)
        new_data.X_scale.append(X_strip_scale)
        new_X_strip = weighted_strip/X_strip_scale
        append_to_h5(new_data.X,new_X_strip)
    f.close()
    #shuffle and save X
    new_X = h5py.File(new_data.X,'r+')['dataset']
    print('shuffling X')
    rng_state = np.random.get_state()
    # shuffle and save snplist
    snps = np.array(pd.read_csv(snplist,delim_whitespace=True).iloc[:,:])
    np.random.shuffle(snps)
    shuffled_snps = pd.DataFrame(data=snps,columns=['CHR','SNP'])
    shuffled_snps.to_csv(snplist+'_shuffled',sep='\t',index=False)
    np.random.set_state(rng_state)
    np.random.shuffle(new_X)
    print('saving shuffled X to '+new_data.X)
    with h5py.File(new_data.X,'r+') as f:
        f['dataset'][:] = new_X
    f.close()
    
    new_data.X_scale = np.concatenate(new_data.X_scale)
    # center weight scale y
    print('processing y')
    y = u.read_chisq_from_ss(dd.y,dd.active_ind)
    new_data.y_offset = dd.mean_y
    centered_y = y - new_data.y_offset
    weighted_y = sqrt_w*centered_y
    new_data.y_scale = np.std(weighted_y)
    new_y = weighted_y/new_data.y_scale
    # shuffle and save y
    np.random.set_state(rng_state)
    np.random.shuffle(new_y)
    new_y_df = pd.DataFrame(data=new_y,columns = ['CHISQ'])
    new_y_df.to_csv(new_data.y,sep='\t',index=False)
    # save X_offset y_offset, X_scale, y_scale to file
    print('writing processed data to file')
    Xoff_df = pd.DataFrame(data=new_data.X_offset,columns=['X_offset'])
    Xoff_df.to_csv(dd.X+'_offset.txt',sep='\t',index=False)
    Xscale_df = pd.DataFrame(data=new_data.X_scale,columns=['X_scale'])
    Xscale_df.to_csv(dd.X+'_scale.txt',sep='\t',index=False)
    yoff_df = pd.DataFrame(data=[new_data.y_offset],columns=['y_offset'])
    yoff_df.to_csv(dd.y+'_offset.txt',sep='\t',index=False)
    yscale_df = pd.DataFrame(data=[new_data.y_scale],columns=['y_scale'])
    yscale_df.to_csv(dd.y+'_scale.txt',sep='\t',index=False)
    # shuffle and save w
    np.random.set_state(rng_state)
    np.random.shuffle(w)
    w_df = pd.DataFrame(data=np.array(w),columns=['WEIGHTS'])
    w_df.to_csv(dd.weights+'_processed.txt',sep='\t',index=False)
    new_data._N = dd.N
    return new_data

def append_to_h5(file_name,array):
    # give h5 file name and an array, append to h5 file column the given array
    c = array.shape[1] # number of columns
    with h5py.File(file_name,'a') as hf:
        d = hf['dataset']
        d.resize(d.shape[1]+c, axis=1)
        d[:,-c:] = array
    hf.close()
    return

def check_processed(data):
    # check if there's a processed version of data
    answer = path.isfile(data.X+'_processed.h5') and path.isfile(data.y+'_processed.txt') and path.isfile(data.X+'_offset.txt') and path.isfile(data.X+'_scale.txt') and path.isfile(data.y+'_offset.txt') and path.isfile(data.y+'_scale.txt')
    return answer

def read_processed(d):
    # construct data object that contains the processed version of d
    dd = data(d.X+'_processed.h5',d.y+'_processed.txt',d.weights,[[0,d.active_len]])
    dd.X_offset = np.array(pd.read_csv(d.X+'_offset.txt',delim_whitespace=True).iloc[:,0])
    dd.X_scale = np.array(pd.read_csv(d.X+'_scale.txt',delim_whitespace=True).iloc[:,0])
    dd.y_offset = np.array(pd.read_csv(d.y+'_offset.txt',delim_whitespace=True).iloc[0,0])
    dd.y_scale = np.array(pd.read_csv(d.y+'_scale.txt',delim_whitespace=True).iloc[0,0])
    dd._N = d.N
    return dd
