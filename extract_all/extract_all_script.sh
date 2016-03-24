#!/bin/sh
echo "Job start taskid: " $SLURM_ARRAY_TASK_ID
source ~/projects/dayabay-data-conversion/setup_make_dataset.sh 
python ~/projects/dayabay-data-conversion/extract_all/extract_background.py $SLURM_ARRAY_TASK_ID
echo "Job end taskid: " $SLURM_ARRAY_TASK_ID
