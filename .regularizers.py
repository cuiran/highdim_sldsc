from keras import regularizers as kr
from keras import backend as K
import numpy as np
import pdb

class L1L2_with_intercept(kr.L1L2):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __call__(self,x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x[:-1]))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x[:-1]))
        return regularization

def l1_with_inter(l=0.01):
    # L1 regularizer without regularizing the last column
    # the last column of the feature represents intercept in this case
    return L1L2_with_intercept(l1=l)
