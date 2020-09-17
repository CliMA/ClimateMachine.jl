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
RUNNAME="hier_gcmdriver_master_micro0-200qct+upperdamp_bulkflux_hsua_diff10_poly3_spongeUandETandRHOthinner28km_refstateT280_no_exp_filter"

# Change if CLIMA and VizCLIMA not saved in $HOME
CLIMA_HOME='/central/groups/esm/lenka/ClimaTests5/ClimateMachine_optimise11.jl'
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
while read -u3 -r line
do
  # Prepare runfile with user-selected parameters
  write_into_runfile_from_list "$line" "$RUNNAME" "$EXPERIMENT" CLIMA_RUNFILE
  # Run climate model
  t_date=$(date +'%m-%d-%y-%T');
  t_start=$(date +%s);
  echo $t_start': Running '$CLIMA_RUNFILE', storing output at '$CLIMA_OUTPUT
  #julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.API.precompile()'
  mpiexec julia --project=$CLIMA_HOME $CLIMA_RUNFILE --experiment=heldsuarez --diagnostics 6shours --monitor-courant-numbers 6shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --init-moisture-profile zero --checkpoint 6shours --surface-flux bulk 
  #--restart-from-num 62
  peak_rss="switched off"
  #~~~~~~
  #julia --project=$CLIMA_HOME $CLIMA_RUNFILE --diagnostics 100steps --monitor-courant-numbers 100steps --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART &
  # Get peak memory usage
  #pid=$!
  #peak_rss="$(get_peak_rss "$pid")";
  #~~~~~~
  # Get wall time
  t_end=$(date +%s);
  t_diff=$((t_end-t_start));
  # Write performance log file
  write_into_perf_log_file $PERF_LOGFILE $CLIMA_RUNFILE $t_date $t_diff $peak_rss;
  sleep 2
  mv $CLIMA_RUNFILE $CLIMA_LOG;
done 3< $EXP_PARAM_FILE;

# This modifies the VIZCLIMA_SCRIPT for this experiment
VIZCLIMA_SCRIPT_BN=$(basename "$VIZCLIMA_SCRIPT")
if [ -d "$CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN" ]; then rm $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN; fi
cp $VIZCLIMA_SCRIPT $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed "s~CLIMA_ANALYSIS =.*~CLIMA_ANALYSIS = \"$CLIMA_ANALYSIS\"~" $VIZCLIMA_SCRIPT > $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed -i "s~CLIMA_NETCDF =.*~CLIMA_NETCDF = \"$CLIMA_NETCDF\"~" $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed -i "s~RUNNAME =.*~RUNNAME = \"$RUNNAME\"~" $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed -i "s~CLIMA_LOGFILE =.*~CLIMA_LOGFILE = \"$PERF_LOGFILE\"~" $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
rm $CLIMA_ANALYSIS"/temp_an";

# Post-process output data from ClimateMachine using VizCLIMA
export GKSwstype=null
julia -e 'import Pkg; Pkg.add("IJulia"); Pkg.add("DelimitedFiles"); Pkg.add("PrettyTables"); Pkg.add("PaddedViews"); Pkg.add("Dates"); Pkg.add("GR")'
julia --project=$VIZCLIMA_HOME -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
VIZCLIMA_LITERATE=$VIZCLIMA_HOME'/src/utils/make_literate.jl'
julia --project=$VIZCLIMA_HOME $VIZCLIMA_LITERATE --input-file $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN --output-dir $CLIMA_ANALYSIS 

mv ${VIZCLIMA_SCRIPT_BN%.jl}.ipynb $CLIMA_ANALYSIS

#jupyter notebook --no-browser --port=0913
