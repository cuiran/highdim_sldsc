import regression as regr
import pdb

def perform_regression(args,data,**kwargs):
    reg_method =args.reg_method
    if reg_method == 'OLS':
        regr.OLS()
    elif reg_method == 'Lasso':
        lasso = regr.Lasso(**kwargs)
        lasso.fit(data)
        reg_obj = lasso
    elif reg_method == 'ElasticNet':
        regr.ElasticNet()
    elif reg_method == 'LassoOLS':
        regr.LassoOLS()
    elif reg_method == 'skLassoCV':
        print('Performing sklearn LassoCV fit...')
        reg_obj = regr.sk_LassoCV(*kwargs)
        reg_obj.fit(data)
    elif reg_method == 'skOLS':
        if args.fit == 'direct':
            print('Performing sklearn OLS direct fit...')
            reg_obj = regr.sk_OLS(**kwargs)
            reg_obj.direct_fit(data)
        elif args.fit == 'manual':
            print('Performing sklearn OLS manual fit...')
            reg_obj = regr.sk_OLS(**kwargs)
            reg_obj.fit(data)
    else:
        raise ValueError('The value of --reg-method is invalid')
    return reg_obj
