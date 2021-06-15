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
using ClimateMachine.VariableTemplates: flattenednames

# to be removed: needed for updating ref state
import ClimateMachine.ODESolvers: update_backward_Euler_solver!
import ClimateMachine.DGMethods: update_auxiliary_state!

# to be removed
using CLIMAParameters#: AbstractEarthParameterSet
struct PlanetParameterSet <: AbstractEarthParameterSet end
get_planet_parameter(p::Symbol) = getproperty(CLIMAParameters.Planet, p)(PlanetParameterSet())

# numerics backends
include("../../backends/backends.jl")
# front-end
include("../utilities/timestepper_abstractions.jl")
include("../utilities/sphere_utils.jl")
include("../utilities/operations.jl")
include("../utilities/callbacks.jl")
include("../simulations.jl")
include("../models/models.jl")
# physics
include("../grid/orientations.jl")
include("../physics/physics.jl")
include("../physics/advection.jl")
include("../physics/thermodynamics.jl")
include("../physics/coriolis.jl")
include("../physics/pressure_force.jl")
include("../physics/gravity.jl")
include("../physics/diffusion.jl")
include("../physics/microphysics.jl")
include("../physics/temperature_profiles.jl")

ClimateMachine.init()