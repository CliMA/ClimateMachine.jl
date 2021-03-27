using MPI
using JLD2
using Test
using Dates
using Printf
using Logging
using StaticArrays
using LinearAlgebra

using ClimateMachine
using ClimateMachine.VTK
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

# imports
import ClimateMachine.VTK: writevtk
import ClimateMachine.Mesh.Grids: polynomialorders

# ×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type AbstractFluid <: BalanceLaw end
struct Fluid <: AbstractFluid end

include("shared_source/domains.jl")
include("shared_source/cubed_shell.jl")
include("shared_source/grids.jl")
include("shared_source/FluidBC.jl")
include("shared_source/abstractions.jl")
include("shared_source/callbacks.jl")
include("shared_source/mass_preserving_filter.jl")
include("plotting/bigfileofstuff.jl")
include("plotting/ScalarFields.jl")
# include("plotting/vizinanigans.jl")
include("plotting/vectorfields.jl")

"""
function coordinates(grid::DiscontinuousSpectralElementGrid)
# Description
Gets the (x,y,z) coordinates corresponding to the grid
# Arguments
- `grid`: DiscontinuousSpectralElementGrid
# Return
- `x, y, z`: views of x, y, z coordinates
"""

function evolve!(simulation, spatial_model; refDat = ())
    Q = simulation.state

    dg = simulation.model
    Ns = polynomialorders(spatial_model)

    if haskey(spatial_model.grid.resolution, :overintegration_order)
        Nover = convention(spatial_model.grid.resolution.overintegration_order, Val(ndims(spatial_model.grid.domain)))
    else
        Nover = (0, 0, 0)
    end

    # only works if Nover > 0
    overintegration_filter!(Q, dg, Ns, Nover)

    function custom_tendency(tendency, x...; kw...)
        dg(tendency, x...; kw...)
        overintegration_filter!(tendency, dg, Ns, Nover)
    end

    t0 = simulation.time.start
    Δt = simulation.timestepper.timestep
    timestepper = simulation.timestepper.method

    odesolver = timestepper(custom_tendency, Q, dt = Δt, t0 = t0)

    cbvector = create_callbacks(simulation, odesolver)

    if isempty(cbvector)
        solve!(Q, odesolver; timeend = simulation.time.finish)
    else
        solve!(
            Q,
            odesolver;
            timeend = simulation.time.finish,
            callbacks = cbvector,
        )
    end

    # Check results against reference if StateCheck callback is used
    # TODO: TB: I don't think this should live within this function
    if any(typeof.(simulation.callbacks) .<: StateCheck)
      check_inds = findall(typeof.(simulation.callbacks) .<: StateCheck)
      @assert length(check_inds) == 1 "Only use one StateCheck in callbacks!"

      ClimateMachine.StateCheck.scprintref(cbvector[check_inds[1]])
      if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[check_inds[1]], refDat)
      end
    end

    return Q
end

function overintegration_filter!(state_array, dgmodel, Ns, Nover)
    if sum(Nover) > 0
        cutoff_order = Ns .+ 1 # yes this is confusing
        # cutoff = ClimateMachine.Mesh.Filters.CutoffFilter(dgmodel.grid, cutoff_order)
        cutoff = MassPreservingCutoffFilter(dgmodel.grid, cutoff_order)
        num_state_prognostic = number_states(dgmodel.balance_law, Prognostic())
        filterstates = 2:4
        filterstates = 1:num_state_prognostic
        ClimateMachine.Mesh.Filters.apply!(
            state_array,
            filterstates,
            dgmodel.grid,
            cutoff,
        )
       
    end

    return nothing
end

function uniform_grid(Ω::AbstractDomain; resolution = (32, 32, 32))
    dims = ndims(Ω)
    resolution = resolution[1:dims]
    uniform = []
    for i in 1:dims
        push!(uniform, range(Ω[i].min, Ω[i].max, length = resolution[i]))
    end
    return Tuple(uniform)
end

# most useful function in Julia
function printcolors()
    for i in 1:255
        printstyled("color = "*string(i)*", ", color = i)
        if i%10==0
            println()
        end
    end
end

function writevtk(filename, simulation)
    dg = simulation.model        
    model = dg.balance_law
    Q = simulation.state
    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)
    return nothing
end
