module Ocean

export AbstractOceanCoupling, Uncoupled, Coupled

abstract type AbstractOceanCoupling end
struct Uncoupled <: AbstractOceanCoupling end
struct Coupled <: AbstractOceanCoupling end

function ocean_init_state! end
function ocean_init_aux! end

function coriolis_parameter end
function kinematic_stress end
function surface_flux end

include("HydrostaticBoussinesq/HydrostaticBoussinesqModel.jl")
include("ShallowWater/ShallowWaterModel.jl")
include("SplitExplicit/SplitExplicitModel.jl")
include("OceanProblems/SimpleBoxProblem.jl")


end
