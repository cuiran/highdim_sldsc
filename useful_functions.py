import numpy as np
import pandas as pd
import h5py


def h5_sum_all(h5file, active_ind):
    # sum all elements in h5file, must have key "dataset"
    d = read_h5(h5file)
    s = np.sum(d[active_ind,:])
    return s

def h5_sum_cols(h5file,active_ind):
    # sum columns in h5file, must have key "dataset"
    d = read_h5(h5file)
    s = np.sum(d[active_ind,:],axis=1)
    return s

def chisq_sum_all(ss_file,active_ind):
    active_chisq = read_chisq_from_ss(ss_file,active_ind)
    return np.sum(active_chisq)

def read_chisq_from_ss(ss_file,active_ind):
    # output active chisq as an ndarray
    ss_df = pd.read_csv(ss_file,delim_whitespace=True)
    chisq = np.array(ss_df['CHISQ'])[active_ind]
    return chisq


def make_chuncks(data,chuncksize=100):
    #TODO finish writing this function
    # chuncking data.X by column, default chunck size is 100
    # output an ordered list of list of ordered indices
    d = read_h5(data.X)
    ncols = d.shape[1]
    if chuncksize > ncols:
        raise ValueError('The chunck size is larger than the size of data')
    else:
        


def read_h5(h5file):
    # the key of the h5file has to be 'dataset' for this function to work properly
    f = h5py.File(h5file,'r')
    return f['dataset']
