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
using ClimateMachine.Atmos: NoReferenceState
using ClimateMachine.MPIStateArrays
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.Orientations
using ClimateMachine.VTK

# to be removed
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
# using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates: flattenednames

# to be removed
using CLIMAParameters#: AbstractEarthParameterSet
#using CLIMAParameters.Planet: grav, R_d, R_v, press_triple, T_triple, cp_d, cp_v, cp_l, cv_d, cv_v, cv_l, LH_v0, e_int_v0, planet_radius, MSLP, Omega, T_0
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

const total_energy = true
const fluctuation_gravity = true

# utils
include("../utilities/operations.jl")
include("../utilities/sphere_utils.jl")
# grid
include("../grid/domains.jl")
include("../grid/grids.jl")
# numerics
include("../models.jl")
include("../numerics/filters.jl")
# physics
include("../physics/physics.jl")
include("../grid/orientations.jl")
include("../physics/advection.jl")
include("../physics/thermodynamics.jl")
include("../physics/coriolis.jl")
include("../physics/gravity.jl")
include("../physics/pressure_force.jl")
include("../physics/diffusion.jl")
include("../physics/microphysics.jl")
include("../physics/temperature_profiles.jl")
# boundary conditions
include("../boundary_conditions/boundary_conditions.jl")
include("../boundary_conditions/bc_first_order.jl")
include("../boundary_conditions/bc_second_order.jl")
include("../boundary_conditions/bc_gradient.jl")
# interface 
include("../balance_law_interface.jl")
include("../esdg_balance_law_interface.jl")
include("../numerics/numerical_volume_fluxes.jl")
include("../numerics/numerical_interface_fluxes.jl")
include("../simulations.jl")
include("../utilities/callbacks.jl")

ClimateMachine.init()