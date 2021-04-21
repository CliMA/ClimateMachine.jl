using Dates
using GaussQuadrature
using JLD2
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.MPIStateArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws
using ClimateMachine.ODESolvers
using ClimateMachine.Orientations
using ClimateMachine.VTK

# utils
include("../utilities/operations.jl")
include("../utilities/sphere_utils.jl")
# grid
include("../grid/domains.jl")
include("../grid/grids.jl")
# numerics
include("../models.jl")
include("../numerics/filters.jl")
include("../numerics/fluxes.jl")
# physics
include("../physics/physics.jl")
include("../grid/orientations.jl")
include("../physics/advection.jl")
include("../physics/thermodynamics.jl")
include("../physics/diffusion.jl")
include("../physics/coriolis.jl")
include("../physics/gravity.jl")
# boundary conditions
include("../boundary_conditions/boundary_conditions.jl")
include("../boundary_conditions/bc_first_order.jl")
include("../boundary_conditions/bc_second_order.jl")
include("../boundary_conditions/bc_gradient.jl")
# interface 
include("../balance_law_interface.jl")
include("../simulations.jl")
include("../utilities/callbacks.jl")

ClimateMachine.init()