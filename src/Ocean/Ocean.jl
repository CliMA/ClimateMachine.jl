module Ocean

export AbstractOceanCoupling, Uncoupled, Coupled

abstract type AbstractOceanCoupling end
struct Uncoupled <: AbstractOceanCoupling end
struct Coupled <: AbstractOceanCoupling end

function coriolis_parameter end

include("HydrostaticBoussinesq/HydrostaticBoussinesqModel.jl")
include("ShallowWater/ShallowWaterModel.jl")
include("SplitExplicit/SplitExplicitModel.jl")
# include("OceanProblems/SimpleBoxProblem.jl")

end
