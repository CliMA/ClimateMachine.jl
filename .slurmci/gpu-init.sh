#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --gres=gpu:1

set -euo pipefail

# to avoid race conditions
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot_gpu"
export PATH="${PATH}:${HOME}/julia-1.2-gpu/bin"
export CUDA_PATH="/lib64"

set -x #echo on

module load cmake/3.10.2 openmpi/3.1.2 cuda/9.1

julia --color=no --project=env/gpu -e 'using Pkg; pkg"instantiate"; pkg"build"; pkg"precompile"'


