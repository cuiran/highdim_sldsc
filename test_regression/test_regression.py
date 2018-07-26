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

def gen_data():
    N = 1000
    p = 6
    beta = np.array([2.0,3.5,0.0,5.8,0.0,1.9]) # 6 features
    beta_df = pd.DataFrame(data=beta,columns=['BETA'])
    beta_df.to_csv('beta.txt',sep='\t',index=False)
    X = np.random.rand(N,p)
    X = preprocessing.scale(X)
    noise_cov = np.identity(N)
    noise = np.random.multivariate_normal(np.zeros(N),noise_cov)
    y = X.dot(beta) + noise
    data_obj = save_make_obj(X,y,noise_cov)
    return data_obj,X,y

def fit():
    data_obj,X,y = gen_data()
    sklasso = linear_model.LassoCV()
    sklasso.fit(X,y)
    klasso = r.Lasso(alpha=sklasso.alpha_)
    klasso.fit(data_obj,data_obj)
    return klasso,sklasso,data_obj


def save_make_obj(X,y,noise_cov):
    dd = d.data('X_processed.h5','y_processed.txt','w_processed.txt',[[0,len(y)]])
    f = h5py.File(dd.X,'w')
    f.create_dataset('dataset',data=X)
    f.close()
    ydf = pd.DataFrame(data=y,columns=['CHISQ'])
    ydf.to_csv('y.txt',sep='\t',index=False)    
    weights = np.diagonal(noise_cov)
    w_df = pd.DataFrame(data=weights,columns=['WEIGHT'])
    w_df.to_csv(dd.weights,sep='\t',index=False)
    dd._active_len = len(y)
    dd._N = 1.0
    dd._mean_X = np.mean(X,axis=0)
    dd._std_X = np.std(X,axis=0)
    dd._mean_y = np.mean(y)
    dd._std_y = np.std(y)
    dd._num_features = X.shape[1]
    dd.X_offset = dd._mean_X
    dd.X_scale = dd._std_X
    dd.y_scale = dd._std_y
    dd.y_offset = dd._mean_y
    y = (y - dd.y_offset)/dd.y_scale
    y_df = pd.DataFrame(data=y,columns=['CHISQ'])
    y_df.to_csv(dd.y,sep='\t',index=False)
    return dd
        

if __name__ == '__main__':
    num_simulations = 5
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
        original_data = d.data('X_processed.h5','y.txt','w_processed.txt',[[0,1000]])
        original_data._N = 1.0
        klasso.evaluate_weighted(original_data)
        klasso_wsse.append(klasso.wsse)
    df = pd.DataFrame(None,columns=['klasso_alpha','sklasso_alpha','klasso_wsse'])
    df['klasso_alpha'] = klasso_alpha
    df['sklasso_alpha'] = sklasso_alpha
    df['klasso_wsse'] = klasso_wsse
    df.to_csv('samealpha_test_alpha_wsse.csv',sep='\t',index=False)
    coef_df = pd.DataFrame(data=klasso_coef,columns=[x for x in range(1,len(klasso_coef[0])+1)])
    coef_df.to_csv('samealpha_test_klassocoef_sim.csv',sep='\t',index=False)
    coef_df = pd.DataFrame(data=sklasso_coef,columns=[x for x in range(1,len(sklasso_coef[0])+1)])
    coef_df.to_csv('samealpha_test_sklassocoef_sim.csv',sep='\t',index=False)
    
