import regression as regr
import data_processing as d
import pdb

def perform_regression(args,data,**kwargs):
    reg_method =args.reg_method
    if reg_method == 'OLS':
        regr.OLS()
    elif reg_method == 'Lasso':
        lasso = regr.Lasso(**kwargs)
        if not d.check_processed(data):
            processed_data = d.preprocess_large(data,args.annot_snplist)
        else:
            print('data already processed')
            processed_data = d.read_processed(data)
        #processed_data = d.preprocess_large(data,args.annot_snplist)
        lasso.fit(processed_data,data)
        reg_obj = lasso
    elif reg_method == 'ElasticNet':
        regr.ElasticNet()
    elif reg_method == 'LassoOLS':
        regr.LassoOLS()
    elif reg_method == 'skLassoCV':
        print('Performing sklearn LassoCV fit...')
        if not d.check_processed(data):
            print('Preprocessing data...')
            processed_data = d.preprocess_large(data)
        else:
            print('Processed data exists')
            processed_data = d.read_processed(data)
        reg_obj = regr.sk_LassoCV()
        reg_obj.fit(processed_data)
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
