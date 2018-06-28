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
#TODO improve algorithm
def get_endpoints(l):
    """
    input a list l. For example [1,2,4,5,6,8,9,12]
    output a list of lists of two elements [[1,3],[4,7],[8,10],[12,13]]
    """
    i = 0
    a = l[0]
    endpoints = []
    start=a
    while i<len(l)-1:
        if a+1 == l[i+1]:
            a+=1
            i+=1
        else:
            end = a+1
            endpoints.append([start,end])
            i+=1
            a=l[i]
            start=a
    endpoints.append([start,l[-1]+1])
    return endpoints

def col_strip(X, list_list_ind):
    # Given a matrix X, and a list of list of start-end indices, cut X by columns of those indcies
    strips = [X[:,list_ind[0]:list_ind[1]] for list_ind in list_list_ind ]
    return strips

#TODO improve algorithm
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

def get_num_SNPs(args):
    ss_df = pd.read_csv(args.sumstats,delim_whitespace=True)
    return ss_df.shape[0]

def make_strips(data, stripsize=1):
    # stripping data.X by column (After chuncking by row), default strip size is 1
    # output an list of lists of endpoints
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

def get_strip_active_X(X,strip,active_ind):
    # assume X is large h5 object, get a strip of X with active rows
    if X.ndim == 1:
        X_strip = X
    else:
        X_strip = X[:,strip[0]:strip[1]]
    active_X_strip = get_active(X_strip,active_ind)
    return active_X_strip

def compute_weighted_chunck(data, ind):
    # for the strip of X given by ind (after chuncking by active indcies), compute weights times chunck
    # return ndarray of weighted chunck
    # ind is given as a list of start and end of columns
    X = read_h5(data.X)
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb.iloc[:,-1])
    if X.ndim == 1:
        active_X_strip = get_active(X,data.active_ind)
    else:
        active_X_strip = get_strip_active_X(X,ind,data.active_ind)
    active_weights = get_active(weights, data.active_ind)
    weighted_chunck = np.transpose(np.multiply(np.transpose(active_X_strip),active_weights)) #double transposing to allow broadcasting
    return weighted_chunck


def compute_wy(data):
    # compute weights times y
    # return ndarray of weighted chisq
    y_tb = pd.read_csv(data.y, delim_whitespace=True)
    chisq = np.array(y_tb['CHISQ'])
    active_chisq = get_active(chisq, data.active_ind)
    weights_tb = pd.read_csv(data.weights, delim_whitespace=True)
    weights = np.array(weights_tb['WEIGHT'])
    weights = get_active(weights, data.active_ind)
    weighted_y = np.multiply(active_chisq,weights)
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

def get_scaled_weighted_Xy(data):
    # this is for processing small data
    X = read_h5(data.X)[:]
    all_w = pd.read_csv(data.weights,delim_whitespace=True).iloc[:,-1]
    active_w = get_active(all_w,data.active_ind)
    scaled_w = stdize_array(active_w)
    wy = compute_wy(data)
    X_col_num = get_X_col_num(X)
    wX = compute_weighted_chunck(data,[[0,X_col_num]])
    if wX.ndim==1:
        wX = wX[np.newaxis].T
    swX = (wX - data.weighted_meanX)/data.weighted_stdX
    swy = (wy - data.weighted_meany)/data.weighted_stdy
    swX_with_intercept = attach_column(swX,scaled_w)
    wX_with_intercept = attach_column(wX,active_w)
    return swX_with_intercept,swy,wX_with_intercept,wy

def attach_column(a,b):
    """ Given ndarray a (2 dimensional) and b a one-dim array,
    attach b to the end of a as a column
    """
    if a.ndim == 1:
        a_with_b = np.concatenate((a[np.newaxis].T,b[np.newaxis].T),axis=1)
    else:
        a_with_b = np.concatenate((a,b[np.newaxis].T),axis=1)
    return a_with_b

def get_X_col_num(X):
    if X.ndim == 1:
        col_num = 1
    else:
        col_num = X.shape[1]
    return col_num

def get_mean_std_w(data):
    weights = pd.read_csv(data.weights,delim_whitespace=True).iloc[:,-1]
    active_weights = get_active(np.array(weights),data.active_ind)
    return np.mean(active_weights),np.std(active_weights)

def get_active_weights(weights_file,active_ind):
    w_df = pd.read_csv(weights_file,delim_whitespace=True)
    start,end = active_ind[0]
    w = w_df.iloc[start:end,-1]
    if len(active_ind) == 1:
        return w
    else:
        for i in active_ind[1:]:
            start,end = i
            w = np.concatenate((w,w_df.iloc[start:end,-1]),axis=0)
        return np.array(w)

def center_scale_Xy(X,y,data):
    centered_X = X - data.weighted_meanX
    new_X = centered_X/data.X_scale
    new_y = y - data.weighted_meany
    if X.ndim == 1:
        new_X = new_X.reshape(-1,1)
    return new_X,new_y

def preprocess_data(data):
    # this is for small data
    # center X and y with weighted mean and scale X with the L2 norm of X - weighted_mean(X)
    # no scaling for y
    # return the centered scaled active X and centered active y
    active_X = get_active(read_h5(data.X),data.active_ind)
    active_y = read_chisq_from_ss(data.y,data.active_ind)
    centered_X,centered_y = center(active_X,active_y,data)
    w = get_active_weights(data.weights,data.active_ind)
    sqrt_w = np.sqrt(w)
    wX,wy = weight_Xy(sqrt_w,centered_X,centered_y)
    new_X,new_y,X_scale,y_scale = scale(wX,wy)
    return new_X,new_y,X_scale,y_scale

def weight_Xy(w,X,y):
    # this is for small data
    # multiply X and y by w
    # w is approximately the square root of estimated variance
    if X.ndim == 1:
        X = X[:,None]
    new_X = w[:,None]*X
    new_y = w*y
    return new_X,new_y

def center(X,y,data):
    centered_X = X - data.mean_X
    centered_y = y - data.mean_y
    return centered_X,centered_y

def scale(X,y):
    # this is for small data
    # return X/std(X), y/std(y)
    X_scale = np.std(X,axis=0)
    y_scale = np.std(y,axis=0)
    return X*(1/X_scale),y*(1/y_scale),X_scale,y_scale


