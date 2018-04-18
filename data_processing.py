import numpy as np
import pandas as pd

class data:
    def __init__(self,args):
        self.ld = args.ld
        self.ss = args.ss
        self.weights = args.weights
        self.annot_SNPs = args.annot_snplist
        self.test_region = args.leave_out
        self.train_ind = []
        self.test_ind = []
    
    def get_traintest_indices(self,args.leave_out):
        # get train and test indices
        

def process(args):
    return data(args)

def match_SNPs(args):
    return
