"""
    Eddy-Diffusivity-Mass-Flux (EDMF)

Module for solving the sub-grid-scale equations
using the Eddy-Diffusivity-Mass-Flux (EDMF) model.
"""
module EDMF

using ...PlanetParameters
using ...MoistThermodynamics
using ..FiniteDifferenceGrids
using ..StateVecs
using ..TriDiagSolvers

include("Utilities.jl")
include("Cases.jl")
include("DirTrees.jl")
include("TurbConvs.jl")
include("AuxiliaryFuncs.jl")
include("InitParams.jl")
include("UpdraftVars.jl")
include("ForcingFuncs.jl")
include("RefState.jl")
include("InitialConditions.jl")
include("PrecomputeVars.jl")
include("ApplyBCs.jl")
include("EDMFFuncs.jl")
include("SurfaceFuncs.jl")
include("SolveTurbConv.jl")
include("ProcessResults.jl")
include("Main.jl")

end # module
