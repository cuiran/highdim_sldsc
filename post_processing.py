import numpy as np

def process(args,reg,data):
    # store coefficient and intercept in file 
    # compute sum of squared error on predicted chisq statistics
    coef_df = pd.DataFrame(reg.coef,columns=['COEF'])
    coef_fname = args.output_folder+'coef.txt'
    coef_df.to_csv(coef_fname,index=False,sep='\t')
    intercept_fname = args.output_folder+'intercept.txt'
    f = open(intercept_fname,'w')
    f.write("INTERCEPT")
    f.write(reg.intercept)
    f.close()
    sse = reg.evaluate(data)
    sse_fname = args.output_folder+'weighted_error.txt'
    fi = open(sse_fname,'w')
    fi.write("WSSE")
    fi.write(sse)
    fi.close()
    return 
