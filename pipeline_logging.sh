#!/bin/bash

# Kill the job if anything fails
set -euo pipefail

# Import helper functions for this script
source ./helper_mod.sh

# User envirnoment setup
RUNNAME='your_run_name'

# Change if CLIMA and VizCLIMA not saved in $HOME
CLIMA_HOME=$HOME'/ClimateMachine.jl'
VIZCLIMA_HOME=$HOME'/VizCLIMA.jl'

# Specify output location
mkdir -p $CLIMA_HOME'/output/'
CLIMA_OUTPUT=$CLIMA_HOME'/output/'$RUNNAME

# Choose CLIMA experiment script and VizCLIMA script
EXPERIMENT=$CLIMA_HOME'/experiments/AtmosGCM/heldsuarez.jl'
VIZCLIMA_SCRIPT=$VIZCLIMA_HOME'/src/scripts/dry_gcm_sensitivity.jl'

# Define a parameter file for experiment
EXP_PARAM_FILE=$CLIMA_HOME'/exp_parameters'

# Prepare directories and julia packages
directory_structure $CLIMA_OUTPUT

julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.instantiate()'

# run each experiment listed in EXP_PARAM_FILE
while read -r line
do
  # Prepare runfile with user-selected parameters
  write_into_runfile_from_list "${line}" "$RUNNAME" "$EXPERIMENT" CLIMA_RUNFILE

  # Run climate model
  t_date=$(date +'%m-%d-%y-%T');
  t_start=$(date +%s);

  echo $t_start': Running '$CLIMA_RUNFILE', storing output at '$CLIMA_OUTPUT
  julia --project=$CLIMA_HOME -e 'using Pkg; Pkg.API.precompile()'
  julia --project=$CLIMA_HOME $CLIMA_RUNFILE --diagnostics 2000steps --monitor-courant-numbers 2000steps --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART &

  # Get peak memory usage
  pid=$!
  peak_rss="$(get_peak_rss "$pid")"

  # Get wall time
  t_end=$(date +%s);
  t_diff=$((t_end-t_start))

  # Write performance log file
  write_into_perf_log_file $PERF_LOGFILE $CLIMA_RUNFILE $t_date $t_diff $peak_rss
  mv $CLIMA_RUNFILE $CLIMA_LOG

done < $EXP_PARAM_FILE;

# This modifies the VIZCLIMA_SCRIPT for this experiment
VIZCLIMA_SCRIPT_BN=$(basename "$VIZCLIMA_SCRIPT")
rm $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
cp $VIZCLIMA_SCRIPT $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed "s~CLIMA_ANALYSIS =.*~CLIMA_ANALYSIS = \"$CLIMA_ANALYSIS\"~" $VIZCLIMA_SCRIPT > $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
sed "s~CLIMA_NETCDF =.*~CLIMA_NETCDF = \"$CLIMA_NETCDF\"~" $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN > $CLIMA_ANALYSIS"/temp_an";
sed "s~RUNNAME =.*~RUNNAME = \"$RUNNAME\"~" $CLIMA_ANALYSIS"/temp_an" > $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
rm $CLIMA_ANALYSIS"/temp_an";

# Post-process output data from ClimateMachine using VizCLIMA
julia --project=$VIZCLIMA_HOME -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'
VIZCLIMA_LITERATE=$VIZCLIMA_HOME'/src/utils/make_literate.jl'
julia --project=$VIZCLIMA_HOME $VIZCLIMA_LITERATE --input-file $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN --output-dir $CLIMA_ANALYSIS

mv ${VIZCLIMA_SCRIPT_BN%.jl}.ipynb $CLIMA_ANALYSIS
