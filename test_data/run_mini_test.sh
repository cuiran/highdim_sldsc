#!/bin/bash

python ../main.py --highdim-sldsc \
 --ld mini_one_annot/annot_ld.h5 \
 --sumstats mini_one_annot/sumstats.txt \
 --weights_ld mini_one_annot/weights_ld. \
 --annot-snplist mini_one_annot/annot_snplist.txt \
 --leave-out chr1 \
 --reg-method Lasso \
 --output-folder ..outputs/mini_one_annot/
