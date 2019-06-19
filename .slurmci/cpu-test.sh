#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core

set -euo pipefail
set -x #echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="${HOME}/.julia-slurmci/"

module load cmake/3.10.2 openmpi/3.1.2

julia --project -e 'push!(LOAD_PATH, "@pkglock"); using PkgLock; PkgLock.instantiate_precompile()'
julia --color=no --project test/runtests.jl
