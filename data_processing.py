import numpy as np
import pandas as pd
import h5py
import pdb
import useful_functions as u
from memory_profiler import profile

class data:
    def __init__(self,X,y,weights,active_ind):
        self.X = X  # file name contain regressor related info
        self.y = y  # file name contain target related info
        self.weights = weights  # file name contain weight
        self.active_ind = active_ind # list of indices of X (also y and weights) that are currently in use
        self._mean_X = None # mean per column of active X
        self._std_X = None  # std percolumn of active X
        self._mean_y = None # mean per col of active y
        self._std_y = None  # std per col of active y
        self._weighted_meanX = None # mean per column of weighted X
        self._weighted_stdX = None
        self._weighted_meany = None
        self._weighted_stdy = None
    


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
    return train_ind,test_ind

def compute_true_w(args,data):
    # concatenate weights in args.weights_ld
    # this is the part of true weights that correct some of the correlated errors
    chr_list = [str(i) for i in range(1,23)]
    weights_fnames = [args.weights_ld+x+'.l2.ldscore.gz' for x in chr_list]
    weights_dfs = [pd.read_csv(x,delim_whitespace=True) for x in weights_fnames]
    to_concat = [df.iloc[:,-1] for df in weights_dfs]
    weights_corr = np.concatenate(to_concat,axis=0)
    weights_corr = np.fmax(weights_corr,1.0) # prevent inverse from being too large
    # compute weights that correct some of the heteroskedasticity
    M = len(data.active_ind)
    M = float(M)
    sum_trainld = h5_sum_all(data.X,data.active_ind)
    sum_trainss = chisq_sum_all(data.y,data.active_ind)
    l = sum_trainld/M
    s = sum_trainss/M
    Ntau_hat = np.divide(s-1,l)
    weights_hetero = 2*((Ntau_hat*h5_sum_cols(data.X,data.active_ind)+1)**2)
    # multiply heteroskedasticity weights with correlation weights
    true_weights = np.multiply(weights_hetero,weights_corr)
    return true_weights

def h5_sum_all(h5_file, active_ind):
    # sum all elements in the h5 file. The file must have key 'dataset'
    f = h5py.File(h5_file,'r')
    d = f['dataset']
    s = np.sum(d[active_ind,:])
    return s

def h5_sum_cols(h5_file,active_ind):
    # sum columns in the h5 file. The file must have key 'dataset'
    f = h5py.File(h5_file,'r')
    d = f['dataset']
    s = np.sum(d[active_ind,:],axis=1)
    return s

def chisq_sum_all(ss_file, active_ind):
    ss_df = pd.read_csv(ss_file,delim_whitespace=True)
    chisq = np.array(ss_df['CHISQ'])
    return np.sum(chisq)

def get_num_SNPs(args):
    snplist_df = pd.read_csv(args.annot_snplist,delim_whitespace=True)
    return snplist_df.shape[0]
