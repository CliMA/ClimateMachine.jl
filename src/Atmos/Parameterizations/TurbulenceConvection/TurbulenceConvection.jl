"""
    TurbulenceConvection

Module for solving the sub-grid-scale equations
using the Eddy-Diffusivity-Mass-Flux (EDMF) model.
"""
module TurbulenceConvection

using Requires
@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
    using .Plots
end


include("Grids/FiniteDifferenceGrids.jl")
include("StateVecs/StateVecs.jl")
include("LinearSolvers/TriDiagSolvers.jl")
include("EDMF/EDMF.jl")

end # module
