import numpy as np
import pandas as pd
import pdb
import sys
sys.path.append('/home/rancui/regularized_sldsc/pyscripts_new/')
import regression as r
from sklearn import linear_model
from sklearn import preprocessing
import data_processing as d
import h5py
import line_profiler

def gen_data():
    N = 100000
    p = 6
    beta = np.array([2.0,3.5,0.0,5.8,0.0,1.9]) # 6 features
    beta_df = pd.DataFrame(data=beta,columns=['BETA'])
    beta_df.to_csv('beta.txt',sep='\t',index=False)
    X = np.random.rand(N,p)
    f = h5py.File('X.h5','w')
    f.create_dataset('dataset',data=X)
    f.close()
    noise = np.random.normal(0.0,1.0,N)
    y = X.dot(beta) + noise
    data_obj = save_make_obj(X,y,np.ones(N))
    return data_obj,X,y

def save_make_obj(X,y,noise_cov):
    dd = d.data('X.h5','y.txt','w.txt',[[0,len(y)]])
    f = h5py.File(dd.X,'w')
    f.create_dataset('dataset',data=X)
    f.close()
    ydf = pd.DataFrame(data=y,columns=['CHISQ'])
    ydf.to_csv('y.txt',sep='\t',index=False)    
    weights = noise_cov
    w_df = pd.DataFrame(data=weights,columns=['WEIGHT'])
    w_df.to_csv(dd.weights,sep='\t',index=False)
    dd._N = 1.0
    return dd
@profile
def fit():
    print('generating data')
    data_obj,X,y = gen_data()
    sklasso = linear_model.LassoCV()
    print('performing sklearn LassoCV fit')
    sklasso.fit(X,y)
    klasso = r.Lasso(alpha=sklasso.alpha_)
    print('processing data')
    processed_data = d.preprocess_large(data_obj)
    print('performing keras lasso fit')
    klasso.fit(processed_data,data_obj)
    return klasso,sklasso,data_obj

if __name__ == '__main__':
    num_simulations = 1
    klasso_coef = []
    sklasso_coef = []
    klasso_alpha = []
    sklasso_alpha = []
    klasso_wsse = []
    for i in range(num_simulations):
        print('simulation number'+str(i))
        klasso,sklasso,dd = fit()
        klasso_coef.append(klasso.coef.flatten())
        sklasso_coef.append(sklasso.coef_.flatten())
        klasso_alpha.append(klasso.alpha)
        sklasso_alpha.append(sklasso.alpha_)
        original_data = d.data('X.h5','y.txt','w.txt',[[0,100000]])
        original_data._N = 1.0
        klasso.evaluate_weighted(original_data)
        klasso_wsse.append(klasso.wsse)
    df = pd.DataFrame(None,columns=['klasso_alpha','sklasso_alpha','klasso_wsse'])
    df['klasso_alpha'] = klasso_alpha
    df['sklasso_alpha'] = sklasso_alpha
    df['klasso_wsse'] = klasso_wsse
    prefix = 'N1M_samealpha_'
    df.to_csv(prefix+'alpha_wsse.csv',sep='\t',index=False)
    coef_df = pd.DataFrame(data=klasso_coef,columns=[x for x in range(1,len(klasso_coef[0])+1)])
    coef_df.to_csv(prefix+'klassocoef_sim.csv',sep='\t',index=False)
    coef_df = pd.DataFrame(data=sklasso_coef,columns=[x for x in range(1,len(sklasso_coef[0])+1)])
    coef_df.to_csv(prefix+'sklassocoef_sim.csv',sep='\t',index=False)
    
