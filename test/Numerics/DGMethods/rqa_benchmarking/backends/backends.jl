# climate machine
include("climate_machine/filters.jl")
include("climate_machine/domains.jl")
include("climate_machine/grids.jl")
include("climate_machine/esdg_balance_law_interface.jl")
include("climate_machine/boundary_conditions.jl")
include("climate_machine/numerical_volume_fluxes.jl")
include("climate_machine/numerical_interface_fluxes.jl")

# climate machine core

abstract type AbstractBackend end

struct ClimateMachineBackend <: AbstractBackend end