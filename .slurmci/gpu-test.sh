#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH --gres=gpu:1

set -euo pipefail
set -x #echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot/${SLURM_JOB_ID}/${HOSTNAME}:$(pwd)/.slurmdepot/common"
export CUDA_PATH="/lib64"

module load cmake/3.10.2 openmpi/3.1.2 cuda/9.1

# we need to build CUDA on each device
# to avoid race conditions we create a separate depot per job
julia --color=no --project=env/gpu -e 'using Pkg; pkg"instantiate"; pkg"precompile"'

julia --color=no --project=env/gpu test/runtests_gpu.jl
