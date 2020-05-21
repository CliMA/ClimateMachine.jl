#!/bin/bash

#SBATCH --job-name=setup_hadgem_gcm_les_central
#SBATCH --output=/groups/esm/asridhar/GCP/setup_hadgem_gcm_les_central.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL

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

julia --project=/home/asridhar/CLIMA -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project=/home/asridhar/CLIMA -e 'using Pkg; Pkg.precompile()'

cat /home/asridhar/CLIMA/Manifest.toml
