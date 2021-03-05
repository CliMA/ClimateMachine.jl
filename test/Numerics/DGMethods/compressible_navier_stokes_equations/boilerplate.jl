using MPI
using JLD2
using Test
using Dates
using Printf
using Logging
using StaticArrays
using LinearAlgebra

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

# ×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
⋅(a::SVector, b::SVector) = StaticArrays.dot(a, b)
⊗(a::SVector, b::SVector) = a * b'

abstract type AbstractFluid <: BalanceLaw end
struct Fluid <: AbstractFluid end

include("shared_source/domains.jl")
include("shared_source/grids.jl")
include("shared_source/FluidBC.jl")
include("shared_source/abstractions.jl")
include("shared_source/callbacks.jl")

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
        Nover = spatial_model.grid.resolution.overintegration_order
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

    ## Check results against reference

    ClimateMachine.StateCheck.scprintref(cbvector[end])
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbvector[end], refDat)
    end

    return Q
end

function visualize(
    simulation::Simulation;
    statenames = [string(i) for i in 1:size(simulation.state)[2]],
    resolution = (32, 32, 32),
)
    a_, statesize, b_ = size(simulation.state)
    mpistate = simulation.state
    grid = simulation.model.grid
    grid_helper = GridHelper(grid)
    r = coordinates(grid)
    states = []
    ϕ = ScalarField(copy(r[1]), grid_helper)
    r = uniform_grid(Ω, resolution = resolution)
    # statesymbol = vars(Q).names[i] # doesn't work for vectors
    for i in 1:statesize
        ϕ .= mpistate[:, i, :]
        ϕnew = ϕ(r...)
        push!(states, ϕnew)
    end
    visualize([states...], statenames = statenames)
end

function overintegration_filter!(state_array, dgmodel, Ns, Nover)
    if sum(Nover) > 0
        cutoff_order = Ns .+ Nover

        cutoff =
            ClimateMachine.Mesh.Filters.CutoffFilter(dgmodel.grid, cutoff_order)
        num_state_prognostic = number_states(dgmodel.balance_law, Prognostic())

        ClimateMachine.Mesh.Filters.apply!(
            state_array,
            1:num_state_prognostic,
            dgmodel.grid,
            cutoff,
        )
    end

    return nothing
end

# initialized on CPU so not problem, but could do kernel launch?
function set_ic!(ϕ, s::Number, _...)
    ϕ .= s
    return nothing
end

function set_ic!(ϕ, s::Function, p, x, y, z)
    _, nstates, _ = size(ϕ)
    @inbounds for i in eachindex(x)
        @inbounds for j in 1:nstates
            ϕʲ = view(ϕ, :, j, :)
            ϕʲ[i] = s(p, x[i], y[i], z[i])[j]
        end
    end
    return nothing
end
