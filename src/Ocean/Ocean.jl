module Ocean

using ..BalanceLaws
using ..Problems

export AbstractOceanModel,
    AbstractOceanProblem,
    AbstractOceanCoupling,
    Uncoupled,
    Coupled,
    AdvectionTerm,
    NonLinearAdvectionTerm,
    InitialConditions

abstract type AbstractOceanModel <: BalanceLaw end
abstract type AbstractOceanProblem <: AbstractProblem end

abstract type AbstractOceanCoupling end
struct Uncoupled <: AbstractOceanCoupling end
struct Coupled <: AbstractOceanCoupling end

abstract type AdvectionTerm end
struct NonLinearAdvectionTerm <: AdvectionTerm end

function ocean_init_state! end
function ocean_init_aux! end
function ocean_boundary_state! end

function coriolis_parameter end
function kinematic_stress end
function surface_flux end

include(joinpath("Domains", "Domains.jl"))
include(joinpath("Fields", "Fields.jl"))

include("OceanBC.jl")

include("HydrostaticBoussinesq/HydrostaticBoussinesq.jl")

using .HydrostaticBoussinesq: HydrostaticBoussinesqModel, Forcing

include("ShallowWater/ShallowWaterModel.jl")
include("SplitExplicit/SplitExplicitModel.jl")
include("SplitExplicit01/SplitExplicitModel.jl")
include("OceanProblems/OceanProblems.jl")

include("SuperModels/SuperModels.jl")

using .OceanProblems: InitialConditions

using .SuperModels:
    SplitExplicitOceanSuperModel,
    HydrostaticBoussinesqSuperModel,
    current_time,
    current_step,
    Δt

include("JLD2Writer.jl")

using .JLD2Writers: JLD2Writer, OutputTimeSeries, write!

end
