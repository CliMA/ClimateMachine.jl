module DGmethods

using MPI
using ..MPIStateArrays
using ..Mesh.Grids
using ..Mesh.Topologies
using StaticArrays
using ..SpaceMethods
using ..VariableTemplates
using DocStringExtensions
using GPUifyLoops

export BalanceLaw, DGModel, State, Grad

include("balancelaw.jl")
include("DGmodel.jl")
include("NumericalFluxes.jl")
include("DGBalanceLawDiscretizations_kernels.jl")

# legacy interface
include("DGBalanceLawDiscretizations.jl")


end
