import h5py
import numpy as np
import pandas as pd
import pdb
import argparse

def check_if_contain(f1, f2, outfile):
    s1 = list(pd.read_csv(f1,delim_whitespace=True)['SNP'])
    s2 = list(pd.read_csv(f2,delim_whitespace=True)['SNP'])
    set1 = set(s1)
    set2 = set(s2)
    answer = set1.issubset(set2)
    f = open(outfile,'w')
    if answer == True:
        f.write(f1+' SNPs is a subset of '+f2+' SNPs')
    else:
        f.write(f1+' SNPs is not a subset of '+f2+' SNPs')
    return answer

def merge_ss(ss_file,outfile):
    asnps = '/home/rancui/data/annot_snplist.txt'
    hm3 = '/home/rancui/data/w_hm3.snplist'
    refdf = pd.read_csv(asnps,delim_whitespace=True,index_col = 0)
    hmdf = pd.read_csv(hm3,delim_whitespace=True,index_col = 0)
    ssdf = pd.read_csv(ss_file,delim_whitespace=True,index_col = 0)
    # merge refdf with hmdf
    refdf = pd.concat([refdf,hmdf],join='inner',axis=1)
    refdf['CHISQ'] = 0.0
    refdf['Z'] = 0.0
    outdf = refdf.join(ssdf,lsuffix='_hm',rsuffix='_ss')
    outdf['A1'] = refdf['A1']
    outdf['A2'] = refdf['A2']
    outdf['Z'] = outdf['Z_hm']+outdf['Z_ss']
    outdf['CHISQ'] = outdf['CHISQ_hm']+outdf['CHISQ_ss']
    outdf['N'] = ssdf.ix[0,'N']
    outdf.drop(labels=['Z_hm','Z_ss','CHISQ_hm','CHISQ_ss'],axis=1,inplace=True)
    outdf.replace(np.inf, np.nan)
    outdf_withna = outdf[['A1','A2','N','CHISQ','Z']]
    outdf = outdf_withna.fillna(0.0)
    outdf.to_csv(outfile,sep='\t')
    return outdf_withna

def merge_weights(w_file):
    asnps = '/home/rancui/data/annot_snplist.txt'
    wdf = pd.read_csv(w_file,delim_whitespace=True,index_col=1)
    adf = pd.read_csv(asnps,delim_whitespace=True,index_col = 0)
    outdf = adf.join(wdf,lsuffix='_hm',rsuffix='_w')
    outdf.fillna(1.0,inplace=True)
    outdf.to_csv('~/data/weights/weights.hm3_yesMHC.csv',sep='\t',index=False)
    return

def adjust_weights(w_file,ss_file,outfile):
    ssadj_df = merge_ss(ss_file,'/tmp/ss_adjusted.csv')
    w = h5py.File(w_file,'r')['dataset'][:]
    ssadj_df['w_mark'] = 1.0 - np.isnan(ssadj_df['CHISQ'])
    wmark = np.array(ssadj_df['w_mark'])
    w_adj = wmark*w
    w_adj[w_adj==0.0] = np.nan
    f = h5py.File(outfile,'w')
    f.create_dataset('dataset',data=w_adj)
    f.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contain', action='store_true')
    parser.add_argument('--file1')
    parser.add_argument('--file2')
    parser.add_argument('--outfile')
    parser.add_argument('--add_to_ssSNPs',action='store_true')
    parser.add_argument('--ss_file')
    parser.add_argument('--weights')
    parser.add_argument('--merge_weights',action='store_true')
    parser.add_argument('--adj_weights',action='store_true')
    args = parser.parse_args()

    if args.contain:
        check_if_contain(args.file1,args.file2,args.outfile)    
    elif args.add_to_ssSNPs:
        merge_ss(args.ss_file,args.outfile)
    elif args.merge_weights:
        merge_weights(args.weights)
    elif args.adj_weights:
        adjust_weights(args.weights,args.ss_file,args.outfile)
