import h5py
import numpy as np
from memory_profiler import profile
import pdb

@profile
def sum_all(ld_file):
    ld_f = h5py.File(ld_file,'r')
    ld = ld_f['dataset']
    active_ind = [x for x in range(400000)]
    s = np.sum(ld[active_ind,:])
    col_sum = np.sum(ld[active_ind,:],axis=1)
    row_sum = np.sum(ld[active_ind,:],axis=0)
    return s,col_sum,row_sum

if __name__ == '__main__':
    sum_all,col_sum,row_sum = sum_all("/home/rancui/regularized_sldsc/data/combined_ld.h5")
    pdb.set_trace()
