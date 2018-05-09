mport numpy as np
import pandas as pd
import h5py

def read_h5(h5file):
    # the key of the h5file has to be 'dataset' for this function to work properly
    f = h5py.File(h5file, 'r')
    return f['dataset']

def h5_sum_all(h5file, active_ind):
    # sum all elements in h5file that are active, must have key "dataset"
    d = read_h5(h5file)
    s = np.sum(raw_chunck(d, active_ind))
    return s


def h5_sum_cols(h5file, active_ind):
    # sum columns in h5file, must have key "dataset"
    d = read_h5(h5file)
    s = np.sum(raw_chunck(d, active_ind), axis=1)
    return s

def col_strip(X, list_list_ind):
    # Given a matrix X, and a list of list of start-end indices, cut X by columns of those indcies
    strips = [X[:,list_ind[0]:list_ind[1]] for list_ind in list_list_ind ]
    return strips


def raw_chunck(array, list_list_ind):
    # Given a matrix X, and a list of list of start-end indices, cut X by raws of those indcies
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
    chisq = raw_chunck(chisq_col, active_ind)
    return chisq


def chisq_sum_all(ss_file, active_ind):
    active_chisq = read_chisq_from_ss(ss_file, active_ind)
    return np.sum(active_chisq)


#NAME CHANGE:  make_chuncks --> make_strips
def make_strips(data, stripsize=100):
    # stripping data.X by column (After chuncking by raw), default strip size is 100
    # output an ordered list of list of ordered indices
    X = read_h5(data.X)
    ncols = X.shape[1]
    if stripsize > ncols:
        raise ValueError('The chunck size is larger than the size of data')
    else:
        inx1 = range(0, ncols, stripsize)+ [ncols]
        inx_list = [inx1[i:i + 2] for i in range(len(inx1) - 1)]
    return inx_list



def compute_weighted_chunck(data, ind):
# for the strip of X given by ind (after chuncking by active indcies), compute weights times chunck
# return ndarray of weighted chunck
# ind is given as a list of start and end of columns

# Reading X and weights
    X = read_h5(data.X)
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb['L2'])

# Chuncking X by raw (Acitve indcies) then stripping by col (ind)
    active_X = raw_chunck(X, data.active_ind)
    active_X_strip = active_X[:,ind[0]:ind[1]]

# Chuncking weights by raw (Acitve indcies) then stripping by col (ind)
    active_weights = raw_chunck(weights, data.active_ind)
    active_weights_inv_sqr = 1/np.sqrt(active_weights)

    weighted_chunck = np.transpose(np.multiply(np.transpose(active_X_strip),active_weights_inv_sqr)) #double transposing to allow broadcasting
    return weighted_chunck


def compute_wy(data):
# compute weights times y
# return ndarray of weighted chisq

# Reading y and chuncking it
    y_tb = pd.read_csv(data.y, delim_whitespace=True)
    y = np.array(y_tb['CHISQ'])
    y = raw_chunck(y, data.active_ind)
# Reading weights and chuncking
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb['L2'])
    weights = raw_chunck(weights, data.active_ind)
# weighting yi*(1/sqrt(wi))
    weights_inv_sqr = 1 / np.sqrt(weights)
    weighted_y = np.multiply(y,weights_inv_sqr)

    return weighted_y

