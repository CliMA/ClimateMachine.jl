#!/bin/bash

#SBATCH --time=1:00:00     # walltime
#SBATCH --ntasks=2         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core

set -euo pipefail

module load julia/1.1.0 cmake/3.10.2 openmpi/3.1.2

julia --color=no --project -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
mpiexec julia --startup-file=no --project src/DGmethods/test/sphere/advection_sphere.jl
