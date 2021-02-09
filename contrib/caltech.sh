#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

if [[ -z "${SLURM_JOB_ID}" ]]; then
    cat << EOF
This script should be used as an sbatch submission script. Usage:

  sbatch [options...] contrib/caltech.sh script [arguments...]

where:
  [options...]: sbatch options. Default are "--ntasks=1 --gres=gpu:1 --time=01:00:00"
  script: is the julia script to run
  [arguments...]: options to be passed to the script
EOF
    exit
fi

set -euo pipefail # kill the job if anything fails
set -x # echo script

module purge
module load julia/1.5.2 hdf5/1.10.1 netcdf-c/4.6.1
if [[ -z "${SLURM_JOB_GPUS}" ]]; then
    # no GPUs allocated to job
    module load openmpi/4.0.4
else
    module load cuda/10.2 openmpi/4.0.4_cuda-10.2
fi

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project -e 'using Pkg; Pkg.precompile()'
mpiexec julia --project "$@"
