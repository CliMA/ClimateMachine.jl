#!/bin/bash

#SBATCH --time=10:00:00                      # max walltime
#SBATCH --ntasks=2                          # processes
##SBATCH --ntasks-per-node=32                # processes per node (32 for Caltech's HPC)
#SBATCH --cpus-per-task=1                    # CPUs per process (1 unless multithreading)
##SBATCH --nodes=4                           # number of nodes
#SBATCH --mem-per-cpu=4G                     # per CPU core (6G for Caltech's HPC)
##SBATCH -J "$CLIMA_SCRIPT_NAME"             # job name; not used
#SBATCH --mail-user="dburov@caltech.edu"     # email address

##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# main

if [[ "${CLIMA_PATH}" == "" ]]; then
  echo "CLIMA_PATH undefined; abort"
  exit 1
fi

if [[ "${CLIMA_SCRIPT_NAME}" == "" ]]; then
  echo "CLIMA_SCRIPT_NAME undefined; abort"
  exit 1
fi

set -euo pipefail

module load julia
module load cmake/3.10.2
module load mpich/3.2.1

mkdir -p "~/clima_runs/"
cd ~/clima_runs/
CUR_DIR=$(date +%j--%H_%M)
mkdir "${CUR_DIR}" || exit 1
cd "${CUR_DIR}"

julia --project=${CLIMA_PATH} -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
# mpiexec julia --project=${CLIMA_PATH} ${CLIMA_SCRIPT_NAME} --update-interval=1
# mpiexec julia --project=${CLIMA_PATH} ${CLIMA_SCRIPT_NAME}
mpiexec julia --project=${CLIMA_PATH} ${CLIMA_SCRIPT_NAME} --vtk-interval=1000


