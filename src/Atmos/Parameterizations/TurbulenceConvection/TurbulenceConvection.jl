"""
    TurbulenceConvection

Module for solving the sub-grid-scale equations
using the Eddy-Diffusivity-Mass-Flux (EDMF) model.
"""
module TurbulenceConvection

include("Grids/FiniteDifferenceGrids.jl")
include("StateVecs/StateVecs.jl")
include("LinearSolvers/TriDiagSolvers.jl")
# include("EDMF/EDMF.jl")

end # module
