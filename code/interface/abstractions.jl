#######
# useful concepts for dispatch
#######

"""
Advection terms

right now really only non-linear or ::Nothing
"""
abstract type AdvectionTerm end
struct NonLinearAdvectionTerm <: AdvectionTerm end

"""
Turbulence Closures

ways to handle drag and diffusion and such
"""
abstract type TurbulenceClosure end

struct LinearDrag{T} <: TurbulenceClosure
    λ::T
end

struct ConstantViscosity{T} <: TurbulenceClosure
    μ::T
    ν::T
    κ::T
    function ConstantViscosity{T}(;
        μ = T(1e-6),   # m²/s
        ν = T(1e-6),   # m²/s
        κ = T(1e-6),   # m²/s
    ) where {T <: AbstractFloat}
        return new{T}(μ, ν, κ)
    end
end

"""
Forcings

ways to add body terms and sources
"""
abstract type Forcing end
abstract type CoriolisForce <: Forcing end

struct fPlaneCoriolis{T} <: CoriolisForce
    fₒ::T
    β::T
    function fPlaneCoriolis{T}(;
        fₒ = T(1e-4), # Hz
        β = T(1e-11), # Hz/m
    ) where {T <: AbstractFloat}
        return new{T}(fₒ, β)
    end
end

struct WindStress{T} <: Forcing
    τₒ::T
    function WindStress{T}(; τₒ = T(1e-4)) where {T <: AbstractFloat}
        return new{T}(τₒ)
    end
end

struct Buoyancy{T} <: Forcing
    α::T # 1/K
    g::T # m/s²
    function Buoyancy{T}(; α = T(2e-4), g = T(10)) where {T <: AbstractFloat}
        return new{T}(α, g)
    end
end

"""
Grouping structs
"""
abstract type AbstractModel end

Base.@kwdef struct SpatialModel{𝒜, 𝒞, 𝒟, ℰ} <: AbstractModel
    balance_law::𝒜
    #physics::ℬ
    numerics::𝒞
    grid::𝒟
    boundary_conditions::ℰ
    #parameters::ℱ
end

polynomialorders(s::SpatialModel) = convention(
    model.grid.resolution.polynomialorder,
    Val(ndims(model.grid.domain)),
)

abstract type ModelPhysics end

Base.@kwdef struct FluidPhysics{A, D, H, C, B} <: ModelPhysics
    advection::A = NonLinearAdvectionTerm()
    dissipation::D = nothing
    hyperdiffusion::H = nothing
    coriolis::C = nothing
    buoyancy::B = nothing
end

abstract type AbstractSimulation end

struct Simulation{𝒜, ℬ, 𝒞, 𝒟, ℰ, ℱ, O, DG,N} <: AbstractSimulation
    model::𝒜
    state::ℬ
    timestepper::𝒞
    initial_conditions::𝒟
    callbacks::ℰ
    simulation_time::ℱ
    odesolver::O
    dgmodel::DG
    name::N
end

function Simulation(;
    model = nothing,
    state = nothing,
    timestepper = nothing,
    initial_conditions = nothing,
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
        initial_conditions,
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