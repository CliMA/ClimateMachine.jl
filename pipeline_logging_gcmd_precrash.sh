#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --job-name=gcmdriver
#SBATCH --time=100:00:00
#SBATCH --output=hier_gcmdriver.out


# Kill the job if anything fails
#set -euo pipefail
set -x # echo script

module purge;
#module load julia/1.4.2 hdf5/1.10.1 netcdf-c/4.6.1 cuda/10.0 openmpi/4.0.3_cuda-10.0 # CUDA-aware MPI
module load julia/1.4.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1


#export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# Import helper functions for this script
source ./helper_mod.sh

# User envirnoment setup
RUNNAME="hier_gcmdriver_bulksfcflux_micro0-200s_maxiter10_hsua_diff"

# Change if CLIMA and VizCLIMA not saved in $HOME
CLIMA_HOME='/central/groups/esm/lenka/ClimaTests5/ClimateMachine_debugall2_clean.jl'
VIZCLIMA_HOME='/central/groups/esm/lenka/VizCLIMA.jl'

# Specify output location
mkdir -p '/central/scratch/elencz/'
CLIMA_OUTPUT='/central/scratch/elencz/output/'$RUNNAME

# Choose CLIMA experiment script and VizCLIMA script
#EXPERIMENT=$CLIMA_HOME'/experiments/AtmosGCM/unstable_radiative_equilibrium_bulk_sfc_flux.jl' #also tested in baroclinic_wave.jl, moist_baroclinic_wave.jl and heldsuarez.jl
EXPERIMENT=$CLIMA_HOME'/experiments/AtmosGCM/GCMDriver/GCMDriver.jl'
VIZCLIMA_SCRIPT=$VIZCLIMA_HOME'/src/scripts/general-gcm-notebook-setup.jl'

# Define a parameter file for experiment 
EXP_PARAM_FILE=$CLIMA_HOME'/exp_parameters'

# Prepare directories and julia packages
directory_structure $CLIMA_OUTPUT

julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate()'
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.precompile()'


# run a --project=$CLIMA_HOME -e ''each experiment listed in EXP_PARAM_FILE
while read -r line
  do
  # Prepare runfile with user-selected parameters
  write_into_runfile_from_list "$line" "$RUNNAME" "$EXPERIMENT" CLIMA_RUNFILE
  # Run climate model
  t_date=$(date +'%m-%d-%y-%T');
  t_start=$(date +%s);
  echo $t_start': Running '$CLIMA_RUNFILE', storing output at '$CLIMA_OUTPUT
done < $EXP_PARAM_FILE;

RESTART_RANK=-1 #-1 if no restart

{ 
  mpiexec julia --project=$CLIMA_HOME $CLIMA_RUNFILE --experiment=heldsuarez_custom --diagnostics 6shours --monitor-courant-numbers 6shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --init-moisture-profile zero --init-base-state heldsuarez --surface-flux bulk --checkpoint 3shours --restart-from-num $RESTART_RANK
} || { 
mpiexec -np 1 julia --project=$CLIMA_HOME assemble_checkpoints.jl $CLIMA_RESTART 'HeldSuarezCustom' 128 $RESTART_RANK && mpiexec -np 1 julia --project=$CLIMA_HOME $CLIMA_RUNFILE --experiment=heldsuarez_custom --diagnostics 6shours --monitor-courant-numbers 6shours --output-dir $CLIMA_NETCDF --checkpoint-dir $CLIMA_RESTART --init-moisture-profile zero --init-base-state heldsuarez --surface-flux bulk --restart-from-num 9999
}  



