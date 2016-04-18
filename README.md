#Scripts to convert raw daya bay root files to hdf5 files.

Usage on Cori:

#MPI4PY version:
sbatch -N 12 mpi_submit_extract_all.sl 

#Job array version:
sbatch submit_extract_all.sl
