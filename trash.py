"""
This file stores discarted codes that are not in use but may be helpful to look at later
"""

def high_dim_sldsc(args):
    if args.standardize:
        # match snps between annotation ld, sumstats and weights, store new files in output-folder, return new file names
        ld_fname,ss_fname,w_fname = d.match_snps(args)
        # split data into training and test according to user specified left out region
        # return array of indices for training and test sets store them in files
        ld_train,ss_train,w_train,ld_test,ss_test,w_test = d.split_data(ld_fname,ss_fname,w_fname,args)        # compute true weights for training data, true weights are the variance of noise
        true_train_weights = d.compute_weights(ld_train,ss_train,w_train,args)
        # create new target by computing chisq-1, store new target
        ttrain_fname = d.new_target(ss_train,args)
        # standardize training ld and store the standardized ld, mean and standard deviation
        stdized_ldtr_fname,mldtr_fname,sldtr_fname = d.standardize_ld(ld_train,args)
        # standardize training ss-1, store it in a new file, together with mean and standard deviation
        stdized_ttr_fname,mttr_fname,sttr_fname = d.standardize_target(ttrain_fname,args)
        # scale train weights so that it has variance 1
        scaled_train_weights = d.scale_train_weights(true_train_weights)
        if args.single-method:
            # run regression with one method
            # store results to file
            coef_fname,bias_fname = r.run_reg_one_method(stdized_ldtr_fname,stdized_ttr_fname,scaled_train_weights,args)
            # use the learned coefs and bias to predict summary statistics on the test data
            p.predict_one_method(coef_fname,bias_fname,mldtr_fname,sldtr_fname,mttr_fname,sttr_fname,args)
        elif args.multi-methods:
            # run regression with multiple methods
            # store results to file
            coef_fname,bias_fname = r.run_reg_multi_methods(stdized_ldtr_fname,stdized_ttr_fname,scaled_train_weights,args)
            p.predict_multi_methods(coef_fname,bias_fname,mldtr_fname,sldtr_fname,mttr_fname,sttr_fname,args)
