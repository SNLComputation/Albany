

#!/bin/bash


module unload cmake cray-netcdf-hdf5parallel python 
module unload intel PrgEnv-intel 
module load PrgEnv-gnu
module unload cray-mpich
module load cray-mpich/7.7.6 
module unload gcc/7.3.0 
module load gcc/8.2.0 
module load boost 
module load cmake python cray-netcdf-hdf5parallel
module list
