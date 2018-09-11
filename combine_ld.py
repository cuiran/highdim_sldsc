import h5py
import argparse
import pandas as pd
import numpy as np
import pdb

def concat_chr(file_prefix,outfile):
    # concat all chromosomes of ldscores to one large h5 file
    files = [file_prefix+str(i)+'.l2.ldscore.gz' for i in range(1,23)]
    dfs = [pd.read_csv(f,delim_whitespace=True) for f in files]
    df = pd.concat(dfs)
    df.to_csv(outfile+'.csv',sep='\t',index=False)
    with h5py.File(outfile,'w') as f:
        f.create_dataset('dataset',data=df.iloc[:,3:])
    f.close()
    return

def concat_single_chr(ld_list,chrom,outfile):
    lddf = pd.read_csv(ld_list,delim_whitespace=True,header=None)
    li = list() # list of dfs
    for i in range(len(lddf)):
        lddir = lddf.iloc[i,0]
        df = pd.read_csv(lddir+chrom+'.l2.ldscore.gz',delim_whitespace=True)
        li.append(df.iloc[:,3:])
        print(lddir)
    allld = np.concatenate(li,axis=1)
    f = h5py.File(outfile,'w')
    f.create_dataset('dataset',data=allld)
    f.close()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concat_chr',action='store_true')
    parser.add_argument('--ldfiles')
    parser.add_argument('--outfile')
    parser.add_argument('--chrom')
    parser.add_argument('--ld_list')
    parser.add_argument('--combine_onechr',action='store_true')
    args = parser.parse_args()

    if args.concat_chr:
        concat_chr(args.ldfiles,args.outfile)
    elif args.combine_onechr:
        concat_single_chr(args.ld_list,args.chrom,args.outfile)
