#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core

set -euo pipefail

# to avoid race conditions
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot_cpu"

module load julia/1.1.0 cmake/3.10.2 openmpi/3.1.2

mpiexec julia --startup-file=no --project $1
