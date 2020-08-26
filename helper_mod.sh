#!/bin/bash

# these are helper functions for the pipeline_logging.sh program

function directory_structure {
  # Directory variables
  CLIMA_RESTART=$1'/restart'
  CLIMA_NETCDF=$1'/netcdf'
  CLIMA_LOG=$1'/log'
  CLIMA_ANALYSIS=$1'/analysis'

  # Create directoy structure
  #rm -rf $1 # delete old output if present
  mkdir -p $1
  mkdir -p $CLIMA_RESTART
  mkdir -p $CLIMA_NETCDF
  mkdir -p $CLIMA_ANALYSIS
  mkdir -p $CLIMA_LOG

  PERF_LOGFILE=$CLIMA_LOG'/experiments_performance_log'
  printf "%-70s %-20s %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-150s \n" "driver" "sim_date" "wall_time" "sim_days" "res_h" "res_v" "dom_height" "CFL" "FT" "GPU" "Mem" "solver_details">> $PERF_LOGFILE
}

function write_into_runfile_from_list {

  line=$1
  runname=$2
  exp_folder=$(dirname "$3")
  default_exp_runfile=$(basename "$3")

  # Get all parameters
  pars=(${line// / });

  # Create name for this particular instance of this experiment
  inst_name=$runname;
  for ((i=0; i<${#pars[@]}; i++))
	do
		inst_name=$inst_name'_'${pars[$i]};
	done
  # remove special chars from new runfile name
  inst_name_clean="$(echo $inst_name | sed 's/[():"]//g')"; # remove special chars
  # Save namelist parameter names and values in separate arrays
  for ((i=0; i<${#pars[@]}; i+=2))
        do
		par_n[$(( $i / 2 ))]=${pars[$i]};
		par_v[$(( $i / 2 ))]=${pars[$(( $i + 1 ))]};
	done

  # Create a copy of the default run file and change to the specified parameters
  new_exp_runfile=${default_exp_runfile%.jl}"_"$inst_name_clean".jl";
  rm -f $exp_folder/$new_exp_runfile;

  # Replace the standard parameters in the run file copy by the modified parameters from $line
  cp $exp_folder/$default_exp_runfile $exp_folder/$new_exp_runfile;
  for ((i=0; i<${#par_n[@]}; i++))
	do
		sed "s/${par_n[$i]} =.*/${par_n[$i]} = ${par_v[$i]}/" $exp_folder/$new_exp_runfile > $exp_folder/run_tmp;
    mv $exp_folder/run_tmp $exp_folder/$new_exp_runfile;
  done

	# Replace default name in standard run file with name of this instance of the experiment $line
	sed "s/exp_name =.*/exp_name = \"$inst_name_clean\"/" $exp_folder/$new_exp_runfile > $exp_folder/run_tmp;
	mv $exp_folder/run_tmp $exp_folder/$new_exp_runfile;

  new_exp_runfile_path=$exp_folder/$new_exp_runfile;

  #Export the new CLIMA_RUNFILE name
  eval $4=$new_exp_runfile_path
}

function extract_vals {
  # From a line in the runfile, extract the value between = and #
  local var=${1}
  local var_=${var%#*}
  local var__=${var_##*=}
  echo $var__ | sed "s/[():]//g"
}

function write_into_perf_log_file {
  # Gather desired performance metrics

  # Imported metrics
  t_date=$3 # date of sim
  t_diff=$4 # wall time
  mem=$5 # rss

  # Metrics extracted from CLIMA_RUNFILE
  sim_time="$(extract_vals "$(grep "n_days =" $CLIMA_RUNFILE)")"
  res_h="$(extract_vals "$(grep "n_horz =" $CLIMA_RUNFILE)")"
  res_v="$(extract_vals "$(grep "n_vert =" $CLIMA_RUNFILE)")"
  domain_height="$(extract_vals "$(grep "domain_height::FT =" $CLIMA_RUNFILE)")"
  FT="$(extract_vals "$(grep "FT =" $CLIMA_RUNFILE)")"
  CFL="$(extract_vals "$(grep "CFL =" $CLIMA_RUNFILE)")"

  solver_method="$(extract_vals "$(grep "solver_method =" $CLIMA_RUNFILE)")"
  implicit_solver="$(extract_vals "$(grep "implicit_solver =" $CLIMA_RUNFILE)")"
  implicit_model="$(extract_vals "$(grep "implicit_model =" $CLIMA_RUNFILE)")"
  splitting_type="$(extract_vals "$(grep "splitting_type =" $CLIMA_RUNFILE)")"
  ode_solver_type="$(extract_vals "$(grep "ode_solver_type =" $CLIMA_RUNFILE)")"
  solver_details=$ode_solver_type/$solver_method/$implicit_solver/$implicit_model/$splitting_type

  gpu="none"

  printf "%-72s %-20s %-15s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-150s \n" $(basename "$CLIMA_RUNFILE") $t_date $t_diff $sim_time $res_h $res_v $domain_height $CFL  $FT $gpu $mem $solver_details >> $1
}

function get_peak_rss {
  # evaluated rss every second (TODO: may need a continuous metric if rss variable)
  pid=$1 peak=0
  while true; do
    sleep 1
    sample="$(ps -o rss= $pid 2> /dev/null)" || break
    let peak='sample > peak ? sample : peak'
  done
  echo $peak
}

function populate_vizclima_script {
  # copy the vizclima script and fill with the run-specific names 
  VIZCLIMA_SCRIPT = $1
  CLIMA_ANALYSIS = $2
  CLIMA_NETCDF = $3
  RUNNAME = $4
  n_plotrows = $5
  n_plotcols = $6
  var_name_1 = $7
  var_name_2 = $8
  var_code_1 = $9
  var_code_2 = ${10}
  
  
  VIZCLIMA_SCRIPT_BN=$(basename "$VIZCLIMA_SCRIPT")
  if [ -d "$CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN" ]; then rm $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN; fi
  cp $VIZCLIMA_SCRIPT $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
  sed "s~CLIMA_ANALYSIS =.*~CLIMA_ANALYSIS = \"$CLIMA_ANALYSIS\"~" $VIZCLIMA_SCRIPT > $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
  sed "s~CLIMA_NETCDF =.*~CLIMA_NETCDF = \"$CLIMA_NETCDF\"~" $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN > "temp_an";
  sed "s~RUNNAME =.*~RUNNAME = \"$RUNNAME\"~" "temp_an" > "temp_an2";
  sed "s~var_sname_1 =.*~var_sname_1 = \"$var_name_1\"~" "temp_an2" > "temp_an3";
  sed "s~var_sname_2 =.*~var_sname_2 = \"$var_name_2\"~" "temp_an3" > "temp_an4";
  sed "s~var_code_1 =.*~var_code_1 = \"$var_code_1\"~" "temp_an4" > "temp_an5";
  sed "s~var_code_2 =.*~var_code_2 = \"$var_code_2\"~" "temp_an5" > "temp_an6";
  sed "s~nrows =.*~nrows = $n_plotrows~" "temp_an6" > "temp_an7";
  sed "s~ncols =.*~ncols = $n_plotcols~" "temp_an7" > $CLIMA_ANALYSIS/$VIZCLIMA_SCRIPT_BN;
  rm '/temp_an*';
  eval ${11}=$VIZCLIMA_SCRIPT_BN
}

