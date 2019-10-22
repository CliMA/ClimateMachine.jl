#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --gres=gpu:1

set -euo pipefail
set -x #echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot/gpu"
export CUDA_PATH="/lib64"
export OPENBLAS_NUM_THREADS=1

module load openmpi/3.1.4 cuda/10.0

# we need to build CUDA on each device
# to avoid race conditions we create a separate depot per job
mpiexec julia --color=no --project=env/gpu $1
