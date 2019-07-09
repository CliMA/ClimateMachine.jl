#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core

set -euxo pipefail

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="${HOME}/.julia-slurmci/"
export OPENBLAS_NUM_THREADS=1

module load cmake/3.10.2 openmpi/3.1.2

julia --project -e 'push!(LOAD_PATH, "@pkglock"); using PkgLock; PkgLock.instantiate_precompile()'
mpiexec julia --color=no --project $1
