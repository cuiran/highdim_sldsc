import pandas as pd
import numpy as np
import h5py
import pdb
import argparse

def ssloss(args):
    if args.baselineLD:
        X = h5py.File(args.X,'r')['dataset'][420505:,-75:]
    else:
        X = h5py.File(args.X,'r')['dataset'][420505:,:]
    y = np.array(pd.read_csv(args.y,delim_whitespace=True)['CHISQ'])[420505:]
    w = np.array(pd.read_csv(args.w,delim_whitespace=True).iloc[:,-1])[420505:]
    coef = np.array(pd.read_csv(args.coef,delim_whitespace=True).iloc[:,0])
    intercept = pd.read_csv(args.intercept,delim_whitespace=True).iloc[0,0]
    N = np.mean(pd.read_csv(args.y,delim_whitespace=True)['N'])
    ypred = N*X.dot(coef)+intercept
    wsse = w.dot((ypred - y)**2)
    wssedf = pd.DataFrame(data=[wsse],columns=['WSSE'])
    wssedf.to_csv(args.out_folder+'_wsse',index=False,sep='\t')
    return

def varbetaloss(args):
    if args.baselineLD:
        ann = h5py.File(args.annot,'r')['dataset'][:,-75:]
    else:
        ann = h5py.File(args.annot,'r')['dataset'][:]
    coef = np.array(pd.read_csv(args.coef,delim_whitespace=True).iloc[:,0])
    varpred = ann.dot(coef)
    truevar = np.array(pd.read_csv(args.truevar,delim_whitespace=True)['VARBETA'])
    sse = np.sum((truevar - varpred)**2)
    ssedf = pd.DataFrame(data=[sse],columns=['SSE'])
    ssedf.to_csv(args.out_folder+'_varbetasse',index=False,sep='\t')
    return
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssloss',action='store_true')
    parser.add_argument('--varbetaloss',action='store_true')
    parser.add_argument('--chr22',action='store_true')
    parser.add_argument('--baselineLD',action='store_true')
    parser.add_argument('--coef')
    parser.add_argument('--intercept')
    parser.add_argument('--X')
    parser.add_argument('--y')
    parser.add_argument('--w')
    parser.add_argument('--annot')
    parser.add_argument('--out_folder')
    parser.add_argument('--truevar')
    args = parser.parse_args()

    if args.ssloss and args.chr22:
        ssloss(args)
    if args.varbetaloss and args.chr22:
        varbetaloss(args) 
