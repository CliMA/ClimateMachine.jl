#!/bin/bash 

#SBATCH --job-name=rtb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00

#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --output=out-rtb

set -euo pipefail # kill the job if anything fails

module load mpich/3.2.1 cmake/3.10.2 cuda/9.1

/home/asridhar/julia-1.3/bin/julia --project=/home/asridhar/CLIMA/env/gpu -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

srun /home/asridhar/julia-1.3/bin/julia --project=/home/asridhar/CLIMA/env/gpu /home/asridhar/CLIMA/test/DGmethods/compressible_Navier_Stokes/rtb_smagorinsky_sgs.jl
