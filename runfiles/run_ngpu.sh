#!/bin/bash
#SBATCH --job-name=ngpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --output=ngpu_test.out
set -euo pipefail # kill the job if anything fails
module load julia/1.3.0
module load cuda/10.0 openmpi/4.0.3_cuda-10.0
export JULIA_MPI_HAS_CUDA=true
export JULIA_MPI_BINARY=system
export CLIMA_HOME=/central/home/asridhar/CLIMA
export outdir=$CLIMA_HOME/checkpoints
outdir=/groups/esm/asridhar/multigpu/hadgem/
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.API.precompile()'
mpirun julia --project=$CLIMA_HOME $CLIMA_HOME/experiments/AtmosLES/hadgem_gcm_les.jl --diagnostics 5smins --vtk 30smins --output-dir $outdir
