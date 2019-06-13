module DGmethods

using MPI
using ..Grids
using ..MPIStateArrays
using StaticArrays
using ..SpaceMethods
using DocStringExtensions
using ..Topologies
using GPUifyLoops

export DGBalanceLaw

include("balancelaw.jl")
include("DGBalanceLawDiscretizations_kernels.jl")
include("NumericalFluxes.jl")
include("dgmodel.jl")



end
