#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --job-name=gcmdriver
#SBATCH --time=20:00:00
#SBATCH --output=model_log_err.out


set SLURM_CONF=/central/groups/esm/lenka/ClimateMachine.jl/slurm.conf
export SLURM_CONF
#JobCompType=jobcomp/filetxt
#JobCompLoc=slurm_job_completion.txt

#AccountingStorageType=accounting_storage/filetxt
#AccountingStorageType=slurm_job_acc.txt


set -euo pipefail # kill the job if anything fails
set -x # echo script

# load correct modules
module purge;
module load julia/1.5.2 hdf5/1.10.1 netcdf-c/4.6.1 openmpi/4.0.1

#export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false

# Import helper functions for this script
source ./helper_mod.sh

# User envirnoment setup
RUNNAME="gcm-sens"

# Specify your home directories and output location on scratch
CLIMA_HOME='/central/groups/esm/lenka/ClimateMachine.jl'
VIZCLIMA_HOME='/central/groups/esm/lenka/VizCLIMA.jl'
SCRATCH_HOME='/central/scratch/elencz/output'

# Choose the ClimateMachine.jl experiment script and VizCLIMA script
EXPERIMENT=$CLIMA_HOME'/experiments/TestCase/baroclinic_wave.jl'
VIZCLIMA_FILE=$VIZCLIMA_HOME'/src/scripts/general-gcm-notebook-setup-multi.jl'

# Define a parameter file for experiment 
EXP_PARAM_FILE=$CLIMA_HOME'/exp_parameters'

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

julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate()'
julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.precompile()'

# Run model for each parameter line of  $EXP_PARAM_FILE
while read -u3 -r line
  do
    write_into_runfile_from_list "$line" "$RUNNAME" "$EXPERIMENT" CLIMA_RUNFILE
    t_start=$(date +%s);
    echo $t_start': Running '$CLIMA_RUNFILE', storing output at '$CLIMA_OUTPUT
    mpiexec julia --project=$CLIMA_HOME $CLIMA_RUNFILE --diagnostics 24shours --monitor-courant-numbers 24shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --sim-time=2592000
    sleep 2
    mv $CLIMA_RUNFILE $CLIMA_LOG;
    cp model_log_err.out $CLIMA_LOG/$(basename $CLIMA_RUNFILE)'.out'
    sacct --format=jobid,elapsed,ncpus,ntasks --state=completed
done 3< $EXP_PARAM_FILE;

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

# View your notebook on a local machine
# on remote terminal (e.g. login2): jupyter notebook --no-browser --port=9999
# on local terminal: ssh -N -f -L 9999:localhost:9999 <yourusername>@login2.hpc.caltech.edu
# in local browser: localhost:9999
# see https://github.com/CliMA/ClimateMachine.jl/wiki/Visualization
