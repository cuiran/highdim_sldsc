import numpy as np
import pandas as pd
import pdb
import useful_functions as u

class data:
    def __init__(self,X,y,weights,active_ind):
        self.X = X  # file name contain regressor related info
        self.y = y  # file name contain target related info
        self.weights = weights  # file name contain weight related info
        self.active_ind = active_ind # indices of X (also y) that are currently in use
        self._mean_X = None # mean per column of active X
        self._std_X = None  # std percolumn of active X
        self._mean_y = None # mean per col of active y
        self._std_y = None  # std per col of active y



def process(args,d):
    # compute y-1 and true weights
    yminus1_fname = store_yminus1(d.y,args.out_folder)
    true_w_fname = store_true_w(d.X,d.y,d.weights,d.active_ind)
    return data(args.ld,yminus1_fname,true_w_fname,d.active_ind)



def match_SNPs(args):
    annot_snps = pd.read_csv(args.annot_snplist,delim_whitespace=True)['SNP'].tolist()
    ss_snps = pd.read_csv(args.sumstats,delim_whitespace=True)['SNP'].tolist()
    if annot_snps==ss_snps:
        return data(args.ld,args.sumstats,args.weights)
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


def store_yminus1(y,folder):
    
