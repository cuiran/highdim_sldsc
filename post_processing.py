import numpy as np
import pandas as pd

def process(args,reg,data):
    # store coefficient and intercept in file 
    # compute sum of squared error on predicted chisq statistics
    coef_df = pd.DataFrame(reg.coef,columns=['COEF'])
    coef_fname = args.output_folder+'coef.txt'
    coef_df.to_csv(coef_fname,index=False,sep='\t')
    intercept_fname = args.output_folder+'intercept.txt'
    f = open(intercept_fname,'w')
    f.write("INTERCEPT \n")
    f.write(str(reg.intercept))
    f.close()
    sse = reg.evaluate_weighted(data)
    sse_fname = args.output_folder+'weighted_error.txt'
    fi = open(sse_fname,'w')
    fi.write("WSSE \n")
    fi.write(str(sse))
    fi.close()
    if hasattr(reg,'alpha'):
        alpha_fname = args.output_folder+'alpha.txt'
        f = open(alpha_fname,'w')
        f.write("ALPHA \n")
        f.write(str(reg.alpha))
        f.close()
    return 
