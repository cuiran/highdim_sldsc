#!/bin/bash

dsub \
   --project "finucane-dp5" \
   --zones "us-east1-b" \
   --logging gs://regularized_sldsc/logging/ \
   --output OUT=gs://regularized_sldsc/output/test_pysub.txt \
   --command "python test_jobsub.py" \
   --wait
