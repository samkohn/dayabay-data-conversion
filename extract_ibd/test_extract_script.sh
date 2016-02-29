#!/bin/sh
echo "Job start taskid: " $TF_TASKID
source setup_make_dataset.sh
python test_extractAD.py $TF_TASKID
echo "Job end taskid: " $TF_TASKID