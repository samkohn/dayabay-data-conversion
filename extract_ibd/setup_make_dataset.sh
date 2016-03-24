#module swap PrgEnv-intel PrgEnv-gnu
module unload python
module unload h5py
module load python/2.7-anaconda
module load root 
export PYTHONPATH=/global/project/projectdirs/das/racah:$PYTHONPATH 
export PYTHONPATH=/usr/common/software/root/6.02.12/lib/root:/usr/common/software/root/6.02.12/bin:$PYTHONPATH

