import pandas as pd
import numpy as np
import h5py
import pdb
import argparse
import sklearn_cheat_hm3snps as reg

def ssloss(args):
    chr_interval = pd.read_csv(args.chr_interval,delim_whitespace=True)
    chrom = int(args.chrom)
    snpstart = chr_interval.iloc[chrom-1,0]
    snpend = chr_interval.iloc[chrom-1,1]
    X = h5py.File(args.X,'r')['dataset'][snpstart:snpend,:]
    ssdf = pd.read_csv(args.y,delim_whitespace=True)
    y = np.array(ssdf['CHISQ'])[snpstart:snpend]
    w_ld = h5py.File(args.w,'r')['dataset'][snpstart:snpend]
    w = reg.compute_weights(w_ld,X,y)
    w[w==0.0]=np.nan
    inv_w = 1/w
    inv_w = np.nan_to_num(inv_w,copy=False)
    coef = np.array(pd.read_csv(args.coef,delim_whitespace=True).iloc[:,0])
    intercept = pd.read_csv(args.intercept,delim_whitespace=True).iloc[0,0]
    N = np.mean(ssdf['N'])
    ypred = N*X.dot(coef)+intercept
    wsse = inv_w.dot((ypred - y)**2)
    persnp_wsse = wsse/np.sum(inv_w)
    worst_wsse = inv_w.dot((np.mean(y) - y)**2)/np.sum(inv_w)
    wwsse = pd.DataFrame(data=[worst_wsse],columns=['mean_est_pwsse'])
    wwsse.to_csv(args.out_folder+'_mean_est_pwsse',index=False,sep='\t')
    persnp = pd.DataFrame(data=[persnp_wsse],columns=['pwsse'])
    persnp.to_csv(args.out_folder+'_pwsse',index=False,sep='\t')
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

def compute_varbeta(args):
    args.annot_prefix
    li = list() # list of varbetas, 22 elts one for each chrom
    for i in range(1,23):
        chrom = str(i)
        print(chrom)
        pdb.set_trace()
        ann = h5py.File(args.annot_prefix+chrom+'.h5','r')['dataset'][:]
        coef = np.array(pd.read_csv(args.coef,delim_whitespace=True).iloc[:,0])
        varbeta = ann.dot(coef)
        li.append(varbeta)
    pdb.set_trace()
    all_var = np.concatenate(li)
    df = pd.DataFrame(data=all_var,columns=['VARBETA'])
    df.to_csv(args.outfile,index=False,sep='\t')
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
    parser.add_argument('--outfile')
    parser.add_argument('--varbeta',action='store_true')
    parser.add_argument('--annot_prefix')
    parser.add_argument('--chrom')
    parser.add_argument('--chr_interval')
    args = parser.parse_args()

    if args.ssloss:
        ssloss(args)
    if args.varbetaloss and args.chr22:
        varbetaloss(args) 
    if args.varbeta and args.chr22:
        compute_varbeta(args)
