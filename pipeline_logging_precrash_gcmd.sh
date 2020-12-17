#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --job-name=gcmdriver
#SBATCH --time=100:00:00
#SBATCH --output=model_log_err.out


# Kill the job if anything fails
#set -euo pipefail
set -x # echo script

module purge;
#module load julia/1.4.2 hdf5/1.10.1 netcdf-c/4.6.1 cuda/10.0 openmpi/4.0.3_cuda-10.0 # CUDA-aware MPI
module load julia/1.5.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1


#export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# Import helper functions for this script
source ./helper_mod.sh

# User envirnoment setup
RUNNAME="gcm-crashdel"

# Change if CLIMA and VizCLIMA not saved in $HOME
CLIMA_HOME='/central/groups/esm/lenka/Clima-del.jl'
VIZCLIMA_HOME='/central/groups/esm/lenka/VizCLIMA.jl'

# Specify output location
mkdir -p '/central/scratch/elencz/'
CLIMA_OUTPUT='/central/scratch/elencz/output/'$RUNNAME

# Choose CLIMA experiment script and VizCLIMA script
#EXPERIMENT=$CLIMA_HOME'/experiments/AtmosGCM/unstable_radiative_equilibrium_bulk_sfc_flux.jl' #also tested in baroclinic_wave.jl, moist_baroclinic_wave.jl and heldsuarez.jl
EXPERIMENT_FILE=$CLIMA_HOME'/experiments/TestCase/baroclinic_wave.jl'
VIZCLIMA_FILE=$VIZCLIMA_HOME'/src/scripts/general-gcm-notebook-setup.jl'

# ------------ No need to change anything below this for the demo ------------
# Prepare directories and julia packages
CLIMA_RESTART=$CLIMA_OUTPUT'/restart'
CLIMA_NETCDF=$CLIMA_OUTPUT'/netcdf'
CLIMA_ANALYSIS=$CLIMA_OUTPUT'/analysis'
CLIMA_LOG=$CLIMA_OUTPUT'/log'
mkdir -p $CLIMA_OUTPUT
mkdir -p $CLIMA_RESTART
mkdir -p $CLIMA_NETCDF
mkdir -p $CLIMA_ANALYSIS
mkdir -p $CLIMA_LOG

julia --project=$CLIMA_HOME -e 'import Pkg; Pkg.add("JLD2")'
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate()'
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.precompile()'

RESTART_RANK=-1 #-1 if no restart

{ 
  mpiexec julia --project=$CLIMA_HOME $EXPERIMENT_FILE --diagnostics 24shours --monitor-courant-numbers 12shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --sim-time=2592000 --checkpoint 1shours --restart-from-num $RESTART_RANK
} || { 
get_restart_no $CLIMA_RESTART .jld2 _num RES_NO  && mpiexec -np 128 julia --project=$CLIMA_HOME $EXPERIMENT_FILE --diagnostics 0.5shours --output-dir $CLIMA_NETCDF --checkpoint-dir $CLIMA_RESTART --restart-from-num $RES_NO --sim-time=2592000
}

# Move VizCLIMA script to the experiment's output location
VIZCLIMA_SCRIPT_BN=$(basename "$VIZCLIMA_FILE")
if [ -d "$CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN" ]; then rm $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN; fi
cp $VIZCLIMA_FILE $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;

# Post-process output data from ClimateMachine using VizCLIMA
export GKSwstype=null
julia -e 'import Pkg; Pkg.add("IJulia"); Pkg.add("DelimitedFiles"); Pkg.add("PrettyTables"); Pkg.add("PaddedViews"); Pkg.add("Dates"); Pkg.add("GR")'
julia --project=$VIZCLIMA_HOME -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
VIZCLIMA_LITERATE=$VIZCLIMA_HOME'/src/utils/make_literate.jl'

cd $CLIMA_ANALYSIS
julia --project=$VIZCLIMA_HOME $VIZCLIMA_LITERATE --input-file $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN --output-dir $CLIMA_ANALYSIS

# setup jupyter notebook (optional)
#jupyter notebook --no-browser --port=9999 &

cd $CLIMA_HOME
# Move log/error file to the experiment's output location

# View your notebook on a local machine
# on remote terminal (e.g. login2): jupyter notebook --no-browser --port=9999
# on local terminal: ssh -N -f -L 9999:localhost:9999 <yourusername>@login2.hpc.caltech.edu
# in local browser: localhost:9999
# see https://github.com/CliMA/ClimateMachine.jl/wiki/Visualization
