import numpy as np
import pandas as pd
import pdb
import useful_functions as u

class data:
    def __init__(self,ld,ss,weights,annot_snplist,leave_out,chrsnp_list):
        self.ld = ld
        self.ss = ss
        self.weights = weights
        self.annot_SNPs = annot_snplist
        self.chrsnp_list = chrsnp_list
        self.test_region = leave_out
        self._train_ind = None
        self._test_ind = None
    
    @property
    def test_ind(self):
        if not self._test_ind: 
            print('Lazy initialization of test_ind')
            chr_list = ['chr'+str(i) for i in range(1,23)]
            if self.test_region in chr_list:
                chrsnp_df = pd.read_csv(self.chrsnp_list,delim_whitespace=True)
                chr_num = int(self.test_region[3:])
                self._test_ind = chrsnp_df.index[chrsnp_df['CHR']==chr_num].tolist()
            else:
                # TODO: the program should be able to take in a file of specified SNPs as leave-out
                print('leave-out functionality is not complete.')
        return self._test_ind

    @property
    def train_ind(self):
        if not self._train_ind:
            print('Lazy initialization of train_ind')
            chr_list = ['chr'+str(i) for i in range(1,23)]
            if self.test_region in chr_list:
                chrsnp_df = pd.read_csv(self.chrsnp_list,delim_whitespace=True)
                all_ind = range(chrsnp_df.shape[0])
                self._train_ind = [x for x in all_ind if x not in self._test_ind]
            else:
                print('leave-out functionality is not complete.')
        return self._train_ind

def process(args,matched_data):
    return data(args.ld,args.sumstats,args.weights,args.annot_snplist,args.leave_out,args.chrsnp_list)

def match_SNPs(args):
    annot_snps = pd.read_csv(args.annot_snplist,delim_whitespace=True)['SNP'].tolist()
    ss_snps = pd.read_csv(args.annot_snplist,delim_whitespace=True)['SNP'].tolist()
    if annot_snps==ss_snps:
        return data(args.ld,args.sumstats,args.weights,args.annot_snplist,args.leave_out,args.chrsnp_list)
    else:
        raise ValueError('--ld and --sumstats must have the same SNPs')
