import argparse
import data_processing as d
import regression as r
import post_processing as p
import pdb
def highdim_sldsc(args):
    """ This function 
        1. process user input
        and return a data object with everything needed for regression analysis. 
        See data_processing.py for attributes of data object.
        2. perform regression on the processed data
        and return a regression object carrying learned coefficients and bias.
        See regression.py for details.
        3. post process the learned coefficients and bias. 
        Use the un-standardized coefficients and bias for prediction on test set.
        Save the un-standardized coefficients and bias to file
    """
    d.match_SNPs(args)
    train_ind,test_ind = d.get_traintest_ind(args)
    train_data = d.data(args.ld,args.sumstats,args.weights,train_ind)
    ready_data = d.process(args,train_data) #change y to y-1, compute true weights on taining data
    reg = r.perform_regression(args,ready_data)
    p.process(args,reg,ready_data)
    return



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--highdim-sldsc',action='store_true',help='run high dimensional stratified ldscore regression')
    parser.add_argument('--ld',help='The .h5 file combining ldscores for all annotations')
    parser.add_argument('--sumstats',help='Text file of formatted summary statistics with columns SNP, A1, A2, N, CHISQ, Z, where CHISQ is Z^2')
    parser.add_argument('--weights',help='The user input weights. Chrom separated "weights.#chr.l2.ldscore.gz" files input as --weights weights.')
    parser.add_argument('--annot-snplist',help='The text file listing all SNPs used to create annotations. This file should contain two columns CHR and SNP. The oder matches the oder of SNPs used in --ld.')
    parser.add_argument('--leave-out',help='Specify the region to leave out, for example, chr22')
    parser.add_argument('--reg-method',help='Name of the method to run regression with. Choose among OLS,Lasso,Lasso+OLS,elnet,grpLasso,skLassoCV,skOLS')
    parser.add_argument('--output-folder',help='Point to a location where the program can store output files')
    args = parser.parse_args()
    
    if args.highdim_sldsc:
        highdim_sldsc(args)

