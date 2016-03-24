#!/bin/bash
#SBATCH -p regular
##SBATCH -t 00:30:00
#SBATCH --qos=premium
#SBATCH -e slurm_outputs/extract_all_%j.err
#SBATCH -o slurm_outputs/extract_all_%j.out 
#SBATCH -J conv-dayabay
source /global/homes/r/racah/projects/dayabay-data-conversion/setup_make_dataset.sh
n_nodes=$SLURM_NNODES
if [ $NERSC_HOST == "cori" ]
then
c_per_node=32
else
c_per_node=24
fi
cores=$(( n_nodes * c_per_node ))
echo "running $cores mpi ranks"
srun -n $cores python /global/homes/r/racah/projects/dayabay-data-conversion/extract_all/mpi_extract_background.py

