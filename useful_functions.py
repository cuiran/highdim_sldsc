import numpy as np
import pandas as pd
import h5py
from sklearn import preprocessing
import pdb

def read_h5(h5file):
    # the key of the h5file has to be 'dataset' for this function to work properly
    f = h5py.File(h5file, 'r')
    return f['dataset']

def h5_sum_all(h5file, active_ind):
    # sum all elements in h5file that are active, must have key "dataset"
    d = read_h5(h5file)
    s = np.sum(get_active(d, active_ind))
    return s


def h5_sum_cols(h5file, active_ind):
    # sum columns in h5file, must have key "dataset"
    d = read_h5(h5file)
    big_chunck = get_active(d,active_ind)
    if big_chunck.ndim == 1:
        return big_chunck
    else:
        s = np.sum(big_chunck, axis=1)
        return s

def col_strip(X, list_list_ind):
    # Given a matrix X, and a list of list of start-end indices, cut X by columns of those indcies
    strips = [X[:,list_ind[0]:list_ind[1]] for list_ind in list_list_ind ]
    return strips


def get_active(array, list_list_ind):
    # Given a matrix X, and a list of list of start-end indices, cut X by rows of those indcies
    if array.ndim == 2:
        chuncks = [array[list_ind[0]:list_ind[1], :] for list_ind in list_list_ind]
    elif array.ndim == 1:
        chuncks = [array[list_ind[0]:list_ind[1]] for list_ind in list_list_ind]
    big_chunck = np.concatenate(chuncks)
    return big_chunck

def read_chisq_from_ss(ss_file, active_ind):
    # output active chisq as an ndarray
    ss_df = pd.read_csv(ss_file, delim_whitespace=True)
    chisq_col = np.array(ss_df['CHISQ'])
    chisq = get_active(chisq_col, active_ind)
    return chisq


def chisq_sum_all(ss_file, active_ind):
    active_chisq = read_chisq_from_ss(ss_file, active_ind)
    return np.sum(active_chisq)


#NAME CHANGE:  make_chuncks --> make_strips
def make_strips(data, stripsize=1):
    # stripping data.X by column (After chuncking by row), default strip size is 100
    # output an ordered list of list of ordered indices
    X = read_h5(data.X)
    if X.ndim == 1:
        ncols = 1
    else:
        ncols = X.shape[1]
    if stripsize > ncols:
        raise ValueError('The strip size is larger than the size of data')
    else:
        inx1 = [x for x in range(0, ncols, stripsize)] + [ncols]
        inx_list = [inx1[i:i + 2] for i in range(len(inx1) - 1)]
    return inx_list



def compute_weighted_chunck(data, ind):
# for the strip of X given by ind (after chuncking by active indcies), compute weights times chunck
# return ndarray of weighted chunck
# ind is given as a list of start and end of columns

# Reading X and weights
    X = read_h5(data.X)
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb['WEIGHT'])

# Chuncking X by row (Acitve indcies) then stripping by col (ind)
    active_X = get_active(X, data.active_ind)
    if active_X.ndim == 1:
        active_X_strip = active_X
    else:
        active_X_strip = active_X[:,ind[0]:ind[1]]

# Chuncking weights by row (Acitve indcies) then stripping by col (ind)
    active_weights = get_active(weights, data.active_ind)
    active_weights_inv_sqr = 1/np.sqrt(active_weights)

    weighted_chunck = np.transpose(np.multiply(np.transpose(active_X_strip),active_weights_inv_sqr)) #double transposing to allow broadcasting
    return weighted_chunck


def compute_wy(data):
# compute weights times y
# return ndarray of weighted chisq

# Reading y and chuncking it
    y_tb = pd.read_csv(data.y, delim_whitespace=True)
    chisq = np.array(y_tb['CHISQ'])
    active_chisq = get_active(chisq, data.active_ind)
# Reading weights and chuncking
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb['WEIGHT'])
    weights = get_active(weights, data.active_ind)
# weighting yi*(1/sqrt(wi))
    weights_inv_sqr = 1 / np.sqrt(weights)
    weighted_y = np.multiply(active_chisq,weights_inv_sqr)

    return weighted_y

def stdize_array(array):
    return preprocessing.scale(array)

def expand_ind(endpoints_list):
    l = []
    for i in endpoints_list:
        l += [x for x in range(i[0],i[1])]
    return l

def convert_to_original_ind(active_ind,new_ind):
    # active_ind is a list of lists of endpoints
    # new_ind is a list of indices, this is the indices of data that's been slices according to active_ind
    # goal is to convert new_ind to the indices based on the original data
    # for example active_ind = [[4,10]], new_ind = [0,1,2], the output should be [4,5,6]
    expanded_active_ind = expand_ind(active_ind)
    new_active_ind = [expanded_active_ind[i] for i in new_ind]
    return new_active_ind
