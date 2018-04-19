import numpy as np
import pandas as pd

class data:
    def __init__(self,args.ld,args.ss,args.weights,args.annot_snplist,args.leave_out,args.chrsnp_list):
        self.ld = args.ld
        self.ss = args.ss
        self.weights = args.weights
        self.annot_SNPs = args.annot_snplist
        self.chrsnp_list = args.chrsnp_list
        self.test_region = args.leave_out
        self._train_ind = None
        self._test_ind = None
    
    @property
    def test_ind(self,args.leave_out,args.chrsnp_list):
        if not _test_ind: 
            chr_list = ['chr'+str(i) for i in range(1,23)]
            if args.leave_out in chr_list:
                chrsnp_df = pd.read_csv(chrsnp_list,delim_whitespace=True)
                chr_num = args.leave_out[3:]
                self._test_ind = chrsnp_df.index[chrsnp_df['CHR']==chr_num].tolist()
            else:
                # TODO: the program should be able to take in a file of specified SNPs as leave-out
                print('leave-out functionality is not complete.')
        return self._test_ind

    @property
    def train_ind(self,args.leave_out,args.chrsnp_list):
        if not _train_ind:
            chr_list = ['chr'+str(i) for i in range(1,23)]
            if args.leave_out in chr_list:
                chrsnp_df = pd.read_csv(chrsnp_list,delim_whitespace=True)
                all_ind = range(chrsnp_df.shape[0])
                self._train_ind = [x for x in all_ind if x not in self._test_ind]
            else:
                print('leave-out functionality is not complete.')
        return self._train_ind

def process(args):
    return data(args.ld,args.ss,args.weights,args.annot_snplist,args.leave_out):

def match_SNPs(args):
    return data(args.ld,args.ss,args.weights,args.annot_snplist,args.leave_out)
