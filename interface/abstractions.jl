#######
# useful concepts for dispatch
#######

"""
Grouping structs
"""
abstract type AbstractModel end

Base.@kwdef struct SpatialModel{𝒜, ℬ, 𝒞} <: AbstractModel
    balance_law::𝒜
    numerics::ℬ
    grid::𝒞
end


polynomialorders(s::SpatialModel) = convention(
    model.grid.resolution.polynomialorder,
    Val(ndims(model.grid.domain)),
)

abstract type AbstractSimulation end

struct Simulation{𝒜, ℬ, 𝒞, 𝒟, ℰ, ℱ, 𝒢, ℋ} <: AbstractSimulation
    model::𝒜
    state::ℬ
    timestepper::𝒞
    callbacks::𝒟
    simulation_time::ℰ
    odesolver::ℱ
    dgmodel::𝒢
    name::ℋ
end

function Simulation(;
    model = nothing,
    state = nothing,
    timestepper = nothing,
    callbacks = nothing,
    simulation_time = nothing,
    odesolver = nothing,
    dgmodel = nothing,
    name = nothing,
)
    # initialize DGModel (rhs)
    dgmodel = DGModel(model) # DGModel --> KernelModel, to be more general? 

    FT = eltype(dgmodel.grid.vgeo)

    # initialize state variables
    if state == nothing
        state = init_ode_state(dgmodel, FT(0); init_on_cpu = true)
    end

    # initialize timestepper
    odesolver = timestepper.method( dgmodel, state; dt = timestepper.timestep, t0 = simulation_time[1] )

    return Simulation(
        model,
        state,
        timestepper,
        callbacks,
        simulation_time,
        odesolver,
        dgmodel,
        name,
    )
end

coordinates(s::Simulation) = coordinates(simulation.model.grid)
polynomialorders(s::Simulation) = polynomialorders(simulation.model.grid)

abstract type AbstractTimestepper end

Base.@kwdef struct TimeStepper{S, T} <: AbstractTimestepper
    method::S
    timestep::T
end

"""
calculate_dt(grid, wavespeed = nothing, diffusivity = nothing, viscocity = nothing, cfl = 0.1)
"""
function calculate_dt(
    grid;
    wavespeed = nothing,
    diffusivity = nothing,
    viscocity = nothing,
    cfl = 1.0,
)
    Δx = min_node_distance(grid)
    Δts = []
    if wavespeed != nothing
        push!(Δts, Δx / wavespeed)
    end
    if diffusivity != nothing
        push!(Δts, Δx^2 / diffusivity)
    end
    if viscocity != nothing
        push!(Δts, Δx^2 / viscocity)
    end
    if Δts == []
        @error("Please provide characteristic speed or diffusivities")
        return nothing
    end
    return cfl * minimum(Δts)
end

abstract type AbstractInitialValueProblem end

Base.@kwdef struct InitialValueProblem{𝒫, ℐ𝒱} <: AbstractInitialValueProblem
    params::𝒫 = nothing
    initial_conditions::ℐ𝒱 = nothing
end

abstract type AbstractBoundaryProblem end

Base.@kwdef struct BoundaryProblem{ℬ𝒞} <: AbstractBoundaryProblem
    boundary_conditions::ℬ𝒞 = nothing
end


#=
function calculate_dt(
    grid::DiscretizedDomain;
    wavespeed = nothing,
    diffusivity = nothing,
    viscocity = nothing,
    cfl = 1.0,
)
    return calculate_dt(
        grid.numerical;
        wavespeed = wavespeed,
        diffusivity = diffusivity,
        viscocity = viscocity,
        cfl = cfl,
    )
end
=#