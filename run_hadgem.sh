#!/bin/bash                                                                                                                    

#SBATCH --job-name=H_CM_I
#SBATCH --output=/home/asridhar/CLIMA/%j_%a.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:30:00
#SBATCH --mem=0
#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL

#SBATCH --array=16

set -euo pipefail 

module load julia/1.3.0
module load openmpi/4.0.3_cuda-10.0 cmake/3.10.2 cuda/10.0 hdf5/1.10.1 netcdf-c/4.6.1

export JULIA_MPI_BINARY=system

julia --project=/home/asridhar/CLIMA/ -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

mpirun julia --project=/home/asridhar/CLIMA /home/asridhar/CLIMA/experiments/AtmosLES/hadgem_gcm_les_amip0K.jl --diagnostics 5smins --output-dir=/home/asridhar/CLIMA/GCP_Round2/Site${SLURM_ARRAY_TASK_ID} --group-id=site${SLURM_ARRAY_TASK_ID} --vtk 30smins
