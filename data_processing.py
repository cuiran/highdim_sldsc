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
        self.train_ind = []
        self.test_ind = []
    
    def get_traintest_indices(self,args.leave_out):
        # get train and test indices
        chr_list = ['chr'+str(i) for i in range(1,23)]
        if args.leave_out in chr_list:
            
        else:
            # TODO: the program should be able to take in a file of specified SNPs to leave out as test
            print('leave-out functionality is not complete')

def process(args):
    return data(args.ld,args.ss,args.weights,args.annot_snplist,args.leave_out):

def match_SNPs(args):
    return data(args.ld,args.ss,args.weights,args.annot_snplist,args.leave_out)
