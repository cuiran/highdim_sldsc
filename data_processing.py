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



def match_SNPs(args):
    annot_snps = pd.read_csv(args.annot_snplist,delim_whitespace=True)['SNP'].tolist()
    ss_snps = pd.read_csv(args.sumstats,delim_whitespace=True)['SNP'].tolist()
    if annot_snps==ss_snps:
        active_ind = [x for x in range(len(ss_snps))]
        return data(args.ld,args.sumstats,args.weights,active_ind)
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
    # concatenate weights in args.weights
    # this is the part of true weights that correct some of the correlated errors
    chr_list = [str(i) for i in range(1,23)]
    weights_fnames = [args.weights+x+'.l2.ldscore.gz' for x in chr_list]
    weights_dfs = [pd.read_csv(x,delim_whitespace=True) for x in weights_fnames]
    to_concat = [df.iloc[:,-1] for df in weights_dfs]
    weights_corr = np.concatenate(to_concat,axis=0)
    weights_corr = np.fmax(weights_corr,1.0) # prevent inverse from being too large
    # compute weights that correct some of the heteroskedasticity
    M = len(data.active_ind)
    M = float(M)
    sum_trainld = sum_all(data.X,data.active_ind)
    sum_trainss = sum_all(data.y,data.active_ind)
    l = sum_trainld/M
    s = sum_trainss/M
    Ntau_hat = np.divide(s-1,l)
    weights_hetero = 2*((Ntau_hat*sum_cols(data.X,data.active_ind)+1)**2)
    # multiply heteroskedasticity weights with correlation weights
    true_weights = np.multiply(weights_hetero,weights_corr)
    return true_weights
