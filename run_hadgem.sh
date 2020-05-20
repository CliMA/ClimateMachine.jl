#!/bin/bash                                                                                                                    

#SBATCH --job-name=site%a_minnodedist
#SBATCH --gres=gpu:1
#SBATCH --time=12:10:00

#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL

#SBATCH --array=17,19,20

#SBATCH --output=/groups/esm/asridhar/GCP/out_%j_site%a_MR5-10.5_V.out
set -euo pipefail 
export JULIA_MPI_BINARY=system
module load julia/1.3.0
module load openmpi/4.0.3_cuda-10.0 cmake/3.10.2 cuda/10.0 hdf5/1.10.1 netcdf-c/4.6.1

export JULIA_MPI_BINARY=system

julia --project=/home/asridhar/CLIMA/ -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

mpirun julia --project=/home/asridhar/CLIMA /home/asridhar/CLIMA/experiments/AtmosLES/hadgem_gcm_les.jl --diagnostics 5smins --output-dir=/central/groups/esm/asridhar/GCP/Site${SLURM_ARRAY_TASK_ID}_MR5-10_V --group-id=site${SLURM_ARRAY_TASK_ID} --vtk 30smins
