import pandas as pd
import numpy as np
from sklearn import linear_model
import h5py
import argparse
import pdb
from sklearn import preprocessing

def get_data(args):
    if args.baselineLD:
        X = h5py.File(args.X,'r')['dataset'][:420505,-75:]
    else:
        X = h5py.File(args.X,'r')['dataset'][:420505,:]
    X_ = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
    y = np.array(pd.read_csv(args.y,delim_whitespace=True)['CHISQ'])[:420505]
    w = np.array(pd.read_csv(args.w,delim_whitespace=True).iloc[:,-1])[:420505]
    wX = w[:,None]*X_
    wy = w[:,None]*y[:,None]
    N = np.mean(pd.read_csv(args.y,delim_whitespace=True)['N'])
    return wX,wy.ravel(),N

def fit(X,y):
    model = linear_model.LassoCV(fit_intercept=False)
    model.fit(X,y)
    return model

def fit_ols(X,y):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X,y)
    return model

def recover(learned_coef,N):
    true_coef = learned_coef[:-1]/N
    true_intercept = learned_coef[-1]
    return true_coef,true_intercept

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
    args = parser.parse_args()

    if args.run_sklearn_fit and args.chr22:
            run(args)
