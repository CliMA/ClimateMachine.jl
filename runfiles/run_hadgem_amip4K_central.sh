#!/bin/bash                                                                                                                    

#SBATCH --job-name=HadGEM24K-A_CliMa
#SBATCH --output=/groups/esm/asridhar/GCP/output_amip4K_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:10:00
#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --array=2-11

set -euo pipefail 
set -x
hostname

module purge
module load julia/1.3.0
module load cuda/10.0 openmpi/4.0.3_cuda-10.0 hdf5/1.10.1 netcdf-c/4.6.1

export JULIA_DEPOT_PATH=/home/asridhar/CLIMA/.julia_depot
export JULIA_MPI_BINARY=system
export OPENBLAS_NUM_THREADS=1
export PATH="/usr/sbin:$PATH"

mpirun julia --project=/home/asridhar/CLIMA /home/asridhar/CLIMA/experiments/AtmosLES/hadgem_gcm_les_amip4K.jl --diagnostics 5smins --output-dir=/central/groups/esm/asridhar/GCP/Site${SLURM_ARRAY_TASK_ID}_amip4K --group-id=site${SLURM_ARRAY_TASK_ID} --vtk 30smins
