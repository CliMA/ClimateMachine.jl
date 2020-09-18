#!/bin/bash
CLIMA_RESTART='/central/scratch/elencz/output/hier_gcmdriver_bulksfcflux_micro0-200s_maxiter10_hsua_diff/restart'
CLIMA_NETCDF='/central/scratch/elencz/output/hier_gcmdriver_bulksfcflux_micro0-200s_maxiter10_hsua_diff/netcdf'


#julia --project assemble_checkpoints.jl $CLIMA_RESTART 'HeldSuarezCustom' 128 -1


mpiexec -np 1 julia --project -e 'using Pkg; Pkg.instantiate()'
mpiexec -np 1 julia --project -e 'using Pkg; Pkg.precompile()'

mpiexec -np 1 julia --project experiments/AtmosGCM/GCMDriver/GCMDriver.jl --experiment=heldsuarez_custom --diagnostics 1shours --monitor-courant-numbers 6shours --output-dir $CLIMA_NETCDF --checkpoint-at-end --checkpoint-dir $CLIMA_RESTART --init-moisture-profile zero --init-base-state heldsuarez --surface-flux bulk --checkpoint 1shours --restart-from-num 9999
