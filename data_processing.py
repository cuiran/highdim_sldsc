import numpy as np
import pandas as pd
import h5py
import pdb
import useful_functions as u
from memory_profiler import profile

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
        weighted_meanX is the mean taken over active rows of weighted X 
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
        self._weighted_meanX = None
        self._weighted_stdX = None
        self._weighted_meany = None 
        self._weighted_stdy = None 
        self._N = None
        self._num_features = None
        self._sum_active_weights = None
        self._X_strips = None
        self._Xstripsize = None

    @property
    def Xstripsize(self):
        return min(self.num_features,100.0)

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
        if not self._mean_X:
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
            self._mean_X = mean.flatten()
            self._std_X = std.flatten()
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
    def sum_active_weights(self):
        active_weights = u.get_active_weights(self.weights,self.active_ind)
        self._sum_active_weights = np.sum(active_weights)
        return self._sum_active_weights
    
    @property
    def X_strips(self):
        # stripping X by column, return a list of lists of endpoints
        self._X_strips = u.make_strips(self,stripsize=self.Xstripsize)
        return self._X_strips

    @property
    def weighted_meanX(self):
        # weighted average of columns of  active X: (\sum w_ix_i)/(\sum w_i)
        w_mean = []
        w = u.get_active_weights(self.weights,self.active_ind)
        for strip in self.X_strips:
            X = u.read_h5(self.X)
            X_strip = u.get_strip_active_X(X,strip,self.active_ind)
            w_mean_strip = np.average(X_strip,weights=w,axis=0)
            w_mean.append(w_mean_strip)
        w_mean = np.array(w_mean)
        self._weighted_meanX = w_mean.flatten()
        return self._weighted_meanX
        
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

    @property
    def weighted_meany(self):
        if not self._weighted_meany: 
            active_w = u.get_active_weights(self.weights,self.active_ind)
            active_y = u.read_chisq_from_ss(self.y,self.active_ind)
            self._weighted_meany = np.average(active_y,weights=active_w)
        return self._weighted_meany

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
    final_train_ind = get_endpoints(train_ind)
    final_test_ind = get_endpoints(test_ind)
    return final_train_ind,final_test_ind

#TODO improve algorithm
def get_endpoints(l):
    """
    input a list l. For example [1,2,4,5,6,8,9,12]
    output a list of lists of two elements [[1,3],[4,7],[8,10],[12,13]]
    """
    i = 0
    a = l[0]
    endpoints = []
    start=a
    while i<len(l)-1:
        if a+1 == l[i+1]:
            a+=1
            i+=1
        else:
            end = a+1
            endpoints.append([start,end])
            i+=1
            a=l[i]
            start=a
    endpoints.append([start,l[-1]+1])
    return endpoints

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
    sum_trainld = u.h5_sum_all(data.X,data.active_ind)
    sum_trainss = u.chisq_sum_all(data.y,data.active_ind)
    l = sum_trainld/M
    s = sum_trainss/M
    Ntau_hat = np.divide(s-1,l)
    weights_hetero = 2*((Ntau_hat*u.h5_sum_cols(data.X,data.active_ind)+1)**2)
    # multiply heteroskedasticity weights with correlation weights
    final_weights = np.multiply(weights_hetero,weights_corr)
    final_weights = np.fmax(final_weights,1.0)
    return final_weights

def weights_processing(args,original_data):
    final_weights = compute_final_w(args,original_data)
    weights_inv = 1/final_weights
    df = pd.DataFrame(data=weights_inv,columns=['WEIGHT'])
    weights_fname = args.output_folder+'final_weights.txt'
    df.to_csv(weights_fname,sep='\t',index=False)
    return weights_fname

def get_num_SNPs(args):
    ss_df = pd.read_csv(args.sumstats,delim_whitespace=True)
    return ss_df.shape[0]
