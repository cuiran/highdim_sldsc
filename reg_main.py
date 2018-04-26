import regression as regr
import pdb

def perform_regression(reg_method,data):
    if reg_method == 'OLS':
        regr.OLS()
    elif reg_method == 'Lasso':
        lasso = regr.Lasso()
        lasso.fit(data)
        reg_obj = lasso
    elif reg_method == 'ElasticNet':
        regr.ElasticNet()
    elif reg_method == 'LassoOLS':
        regr.LassoOLS()
    else:
        raise ValueError('The value of --reg-method is invalid')
    return reg_obj
