#!/bin/bash
#SBATCH -p shared
#SBATCH -n 1
#SBATCH -t 01:00:00
###SBATCH --qos=premium
###SBATCH --array 1001-10000
#SBATCH -e slurm_outputs/extract_all_%j.err
#SBATCH -o slurm_outputs/extract_all_%j.out 
#module load taskfarmer
#tf -t 100000 -n 10 -e serial-3.err -o serial-3.out ~/projects/dayabay-data-conversion/extract_all/extract_all_script.sh
~/projects/dayabay-data-conversion/extract_all/extract_all_script.sh
