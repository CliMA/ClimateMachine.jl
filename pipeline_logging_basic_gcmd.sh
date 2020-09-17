#!/bin/bash

#SBATCH --ntasks=32
#SBATCH --job-name=gcmdriver
#SBATCH --time=20:00:00
#SBATCH --output=model_log_err.out

set -euo pipefail # kill the job if anything fails
set -x # echo script

# load correct modules
module purge;
module load julia/1.4.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

#export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# User envirnoment setup
RUNNAME="demo-gcm-run"

# Specify your home directories and output location on scratch
CLIMA_HOME='/central/groups/esm/lenka/ClimateMachine.jl'
VIZCLIMA_HOME='/central/groups/esm/lenka/VizCLIMA.jl'
SCRATCH_HOME='/central/scratch/elencz/output'

# Choose the ClimateMachine.jl experiment script and VizCLIMA script
EXPERIMENT_FILE=$CLIMA_HOME'/experiments/AtmosGCM/GCMDriver/GCMDriver.jl'
VIZCLIMA_FILE=$VIZCLIMA_HOME'/src/scripts/general-gcm-notebook-setup.jl'

# ------------ No need to change anything below this for the demo ------------
# Setup directory structure
CLIMA_OUTPUT=$SCRATCH_HOME'/'$RUNNAME
CLIMA_RESTART=$CLIMA_OUTPUT'/restart'
CLIMA_NETCDF=$CLIMA_OUTPUT'/netcdf'
CLIMA_ANALYSIS=$CLIMA_OUTPUT'/analysis'
CLIMA_LOG=$CLIMA_OUTPUT'/log'
mkdir -p $CLIMA_OUTPUT
mkdir -p $CLIMA_RESTART
mkdir -p $CLIMA_NETCDF
mkdir -p $CLIMA_ANALYSIS
mkdir -p $CLIMA_LOG

# Run model
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate()'
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.precompile()'

mpiexec julia --project=$CLIMA_HOME $EXPERIMENT_FILE --experiment=heldsuarez --diagnostics 0.5shours --monitor-courant-numbers 6shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --init-moisture-profile zero --checkpoint 6shours

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
mv model_log_err.out $CLIMA_LOG

# View your notebook on a local machine
# on remote terminal (e.g. login2): jupyter notebook --no-browser --port=9999
# on local terminal: ssh -N -f -L 9999:localhost:9999 <yourusername>@login2.hpc.caltech.edu
# in local browser: localhost:9999
# see https://github.com/CliMA/ClimateMachine.jl/wiki/Visualization
