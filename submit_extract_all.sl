#!/bin/bash
#SBATCH -p shared
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --array 0-2500
#SBATCH -e slurm_outputs/extract_all_%j.err
#SBATCH -o slurm_outputs/extract_all_%j.out 
~/projects/dayabay-data-conversion/extract_all/extract_all_script.sh
