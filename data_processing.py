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

    @property
    def mean_X(self):
        if not self._mean_X:
            d = u.read_h5(data.X)
            sum_active_rows = 0
            for i in self.active_ind:
                start,end = i
                sum_chunck = np.sum(d[start:end,:],axis=0)
                sum_active_rows += sum_chunck
            self._mean_X = np.divide(sum_active_rows,self.active_len)
        return self._mean_X

    @property
    def std_X(self):
        if not self._std_X:
            d = u.read_h5(data.X)
            sum_sqdiff = 0
            for i in self.active_ind:
                start,end = i
                sum_sqdiff_chunck = np.sum((d[start:end,:] - self.mean_X)**2,axis=0)
                sum_sqdiff += sum_sqdiff_chunck
            self._std_X = np.sqrt(np.divide(sum_sqdiff,self.active_len))
        return self._std_X
    
    @property
    def weighted_meanX(self):
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
            self._weighted_meanX = np.divide(sum_active_weighted_rows,self.active_len)
        return self._weighted_meanX

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
            self._weighted_stdX = np.sqrt(np.divide(sum_sqwdiff,self.active_len))
        return self._weighted_stdX

    @property
    def weighted_meany(self):
        if not self._weighted_meany:
            full_weights = np.array(pd.read_csv(self.weights,delim_whitespace=True).iloc[:,-1])
            full_y = np.array(pd.read_csv(self.y,delim_whitespace=True)['CHISQ'])
            active_weights = u.get_active(full_weights,self.active_ind)
            to_multiply_w = np.divide(1,np.sqrt(active_weights))
            active_y = u.get_active(full_y,self.active_ind)
            wy = to_multiply_w*active_y
            self._weighted_meany = np.mean(wy)
            self._weighted_stdy = np.std(wy)
        return self._weighted_meany

    @property
    def weighted_stdy(self):
        if not self._weighted_stdy:
            full_weights = np.array(pd.read_csv(self.weights,delim_whitespace=True).iloc[:,-1])
            full_y = np.array(pd.read_csv(self.y,delim_whitespace=True)['CHISQ'])
            active_weights = u.get_active(full_weights,self.active_ind)
            to_multiply_w = np.divide(1,np.sqrt(active_weights))
            active_y = u.get_active(full_y,self.active_ind)
            wy = to_multiply_w*active_y
            self._weighted_meany = np.mean(wy)
            self._weighted_stdy = np.std(wy)
        return self._weighted_stdy

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
    df = pd.DataFrame(data=final_weights,columns=['WEIGHT'])
    weights_fname = args.output_folder+'final_weights.txt'
    df.to_csv(weights_fname,sep='\t',index=False)
    return weights_fname

def get_num_SNPs(args):
    ss_df = pd.read_csv(args.sumstats,delim_whitespace=True)
    return ss_df.shape[0]
