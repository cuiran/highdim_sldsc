import regression as regr
import pdb

def perform_regression(args,data):
    if args.reg_method == 'OLS':
        regr.OLS()
    elif args.reg_method == 'Lasso':
        lasso = regr.Lasso()
        lasso.fit(data)
        reg_obj = lasso
    elif args.reg_method == 'ElasticNet':
        regr.ElasticNet()
    elif args.reg_method == 'LassoOLS':
        regr.LassoOLS()
    else:
        raise ValueError('The value of --reg-method is invalid')
    return reg_obj
