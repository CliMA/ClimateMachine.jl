module SplitExplicit

using StaticArrays

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


include("VerticalIntegralModel.jl")
include("Communication.jl")

end
