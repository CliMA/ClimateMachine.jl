module SplitExplicit

using StaticArrays

using ..Ocean
using ..HydrostaticBoussinesq
using ..ShallowWater

using ...VariableTemplates
using ...MPIStateArrays
using ...Mesh.Geometry
using ...DGMethods
using ...BalanceLaws

import ...BalanceLaws:
    initialize_states!,
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

HBModel = HydrostaticBoussinesqModel
SWModel = ShallowWaterModel

function initialize_states!(
    ::HBModel{C},
    ::SWModel{C},
    _...,
) where {C <: Uncoupled}
    return nothing
end
function tendency_from_slow_to_fast!(
    ::HBModel{C},
    ::SWModel{C},
    _...,
) where {C <: Uncoupled}
    return nothing
end
function cummulate_fast_solution!(
    ::HBModel{C},
    ::SWModel{C},
    _...,
) where {C <: Uncoupled}
    return nothing
end
function reconcile_from_fast_to_slow!(
    ::HBModel{C},
    ::SWModel{C},
    _...,
) where {C <: Uncoupled}
    return nothing
end

include("VerticalIntegralModel.jl")
include("Communication.jl")
include("ShallowWaterCoupling.jl")
include("HydrostaticBoussinesqCoupling.jl")

end
