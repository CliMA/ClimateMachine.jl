module DGmethods

using MPI
using ..MPIStateArrays
using ..Mesh.Grids
using ..Mesh.Topologies
using StaticArrays
using ..VariableTemplates
using DocStringExtensions
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

export BalanceLaw,
    DGModel, init_ode_state, restart_ode_state, restart_auxiliary_state

include("balance_law.jl")
include("NumericalFluxes.jl")
include("DGmodel.jl")
include("DGmodel_kernels.jl")

end
