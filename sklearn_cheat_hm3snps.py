import pandas as pd
import numpy as np
from sklearn import linear_model
import h5py
import argparse
import pdb
from sklearn import preprocessing


def get_data(args):
    ssdf = pd.read_csv(args.y,delim_whitespace=True)
    chr_interval = pd.read_csv(args.chr_interval,delim_whitespace=True)
    if args.chrom == '22':
        snpstart = 0
        snpend = chr_interval.iloc[21,0]
        X = h5py.File(args.X,'r')['dataset'][snpstart:snpend,:]
        y = np.array(ssdf['CHISQ'])[snpstart:snpend]
        w_ld = h5py.File(args.w,'r')['dataset'][snpstart:snpend]
    elif args.chrom == '1':
        snpstart = chr_interval.iloc[1,0]
        snpend = chr_interval.iloc[21,1]
        X = h5py.File(args.X,'r')['dataset'][snpstart:snpend,:]
        y = np.array(ssdf['CHISQ'])[snpstart:snpend]
        w_ld = h5py.File(args.w,'r')['dataset'][snpstart:snpend]
    else:
        chrom = int(args.chrom)
        s1 = 0
        e1 = chr_interval.iloc[chrom-1,0]
        s2 = chr_interval.iloc[chrom-1,1]
        e2 = chr_interval.iloc[21,1]
        X1 = h5py.File(args.X,'r')['dataset'][s1:e1,:]
        X2 = h5py.File(args.X,'r')['dataset'][s2:e2,:]
        X = np.concatenate((X1,X2),axis=0)
        y1 = np.array(ssdf['CHISQ'])[s1:e1]
        y2 = np.array(ssdf['CHISQ'])[s2:e2]
        y = np.concatenate((y1,y2),axis=0)
        w1 = h5py.File(args.w,'r')['dataset'][s1:e1]
        w2 = h5py.File(args.w,'r')['dataset'][s2:e2]
        w_ld = np.concatenate((w1,w2),axis=0)
        
    X_ = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    w = compute_weights(w_ld,X,y)
    w[w==0.0] = np.nan
    sqrt_inv_w = np.sqrt(1/w)
    sqrt_inv_w = np.nan_to_num(sqrt_inv_w,copy=False)
    wX = sqrt_inv_w[:,None]*X
    wy = sqrt_inv_w[:,None]*y[:,None]
    N = np.mean(ssdf['N'])
    return wX,wy.ravel(),N

def compute_weights(w_ld,a_ld,ss):
    w1 = np.array([max(el,1.0) for el in w_ld]) # make everything above 1.0 to prevent inverse from being too large
    M = a_ld.shape[0]
    sum_ld = np.sum(a_ld)
    sum_ss = np.sum(ss)
    l = sum_ld/M
    s = sum_ss/M
    Ntau_hat = (s-1)/l
    sum_ld_col = np.sum(a_ld,axis=1)
    w_hetero = 2*((Ntau_hat*sum_ld_col + 1)**2)
    w1 = np.nan_to_num(w1,copy=False)
    w_hetero = np.fmax(w_hetero,1.0)
    w = np.multiply(w_hetero,w1)
    return w

def fit(X,y):
    model = linear_model.LassoCV(fit_intercept=True,max_iter=2000)
    model.fit(X,y)
    return model

def fit_ols(X,y):
    model = linear_model.LinearRegression(fit_intercept=True)
    model.fit(X,y)
    return model


def recover(learned_coef,N):
    true_coef = learned_coef[:-1]/N
    c = learned_coef[-1]
    return true_coef,c

def run(args):
    X,y,N = get_data(args)
    if args.OLS:
        model = fit_ols(X,y)
    else:
        model = fit(X,y)
        alphadf = pd.DataFrame(data=[model.alpha_],columns=['ALPHA'])
        alphadf.to_csv(args.out_folder+'_alpha',index=False,sep='\t')
    tau,c = recover(model.coef_,N)
    taudf = pd.DataFrame(data=tau,columns=['TAU'])
    taudf.to_csv(args.out_folder+'_coef',index=False,sep='\t')
    cdf = pd.DataFrame(data=[c],columns=['INTERCEPT'])
    cdf.to_csv(args.out_folder+'_intercept',index=False,sep='\t')
    return 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_sklearn_fit',action='store_true')
    parser.add_argument('--chr22',action='store_true')
    parser.add_argument('--baselineLD',action='store_true')
    parser.add_argument('--X')
    parser.add_argument('--y')
    parser.add_argument('--w')
    parser.add_argument('--OLS',action='store_true')
    parser.add_argument('--out_folder')
    parser.add_argument('--chr_interval')
    parser.add_argument('--chrom')
    args = parser.parse_args()

    if args.run_sklearn_fit:
            run(args)
