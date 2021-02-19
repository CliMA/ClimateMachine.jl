using ClimateMachine, MPI
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods

using ClimateMachine.ODESolvers

using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, Omega, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

ClimateMachine.init()
FT = Float64

# Shared functions
include("domains.jl")
include("interface.jl")
include("abstractions.jl")

# Main balance law and its components
include("hyperdiffusion_model.jl") # specific model component 
include("test_model.jl") # umbrella model: TestEquations


# BL and problem for this
hyperdiffusion = HyperDiffusion(
                        HyperDiffusionCubedSphereProblem{FT}((;
                            τ = 1,
                            l = 7,
                            m = 4,
                            )...,
                        )
                    )       


# Domain
Ω = AtmosDomain(radius = planet_radius(param_set), height = 30e3)

# Grid
nelem = (;horizontal = 8, vertical = 4)
polynomialorder = (;horizontal = 5, vertical = 5)
grid = DiscontinuousSpectralElementGrid(Ω, nelem, polynomialorder)
dx = min_node_distance(grid, HorizontalDirection())

# Numerics-specific options
numerics = (; flux = CentralNumericalFluxFirstOrder() ) # add  , overintegration = 1

# Timestepping
Δt_ = Δt(hyperdiffusion.problem, dx)
timestepper = TimeStepper(method = LSRK54CarpenterKennedy, timestep = Δt_ )
start_time, end_time = ( 0  , 2Δt_ )

# Callbacks (TODO)
callbacks = ()

# Specify RHS terms and any useful parameters
balance_law = TestEquations{FT}(
        Ω;
        advection = nothing, # adv 
        turbulence = nothing, # turb
        hyperdiffusion = hyperdiffusion, # hyper
        coriolis = nothing, # cori
        params = nothing,
    )

# Collect all spatial info of the model 
model = SpatialModel( 
    balance_law = balance_law,
    numerics = numerics,
    grid = grid,
    boundary_conditions = nothing,
)

# Initialize simulation with time info
simulation = Simulation(
    model = model,
    timestepper = timestepper,
    callbacks = callbacks,
    simulation_time = (start_time, end_time),
)

# Run the model
solve!( 
    simulation.state, 
    simulation.odesolver; 
    timeend = end_time,
    callbacks = callbacks,
)


#=

abstract type AbstractSimulation end

struct Simulation{𝒜, ℬ, 𝒞, 𝒟, ℰ, ℱ} <: AbstractSimulation
    model::𝒜
    state::ℬ
    timestepper::𝒞
    initial_conditions::𝒟
    callbacks::ℰ
    simulation_time::ℱ
end

function Simulation(;
    model = nothing,
    state = nothing,
    timestepper = nothing,
    initial_conditions = nothing,
    callbacks = nothing,
    simulation_time = nothing,
)
    model = DGModel(model)

    FT = eltype(model.grid.vgeo)

    if state == nothing
        state = init_ode_state(model, FT(0); init_on_cpu = true)
    end
    # model = (discrete = dgmodel, spatial = model)
    return Simulation(
        model,
        state,
        timestepper,
        initial_conditions,
        callbacks,
        simulation_time,
    )
end
________



function DGModel(model::SpatialModel{BL}) where {BL <: AbstractFluid3D}
    params = model.parameters
    physics = model.physics

    Lˣ, Lʸ, Lᶻ = length(model.grid.domain)
    bcs = get_boundary_conditions(model)
    FT = eltype(model.grid.numerical.vgeo)
    balance_law = CNSE3D{FT}(
        (Lˣ, Lʸ, Lᶻ),
        physics.advection,
        physics.dissipation,
        physics.coriolis,
        physics.buoyancy,
        bcs,
        ρₒ = params.ρₒ,
        cₛ = params.cₛ,
    )

    numerical_flux_first_order = model.numerics.flux # should be a function

    rhs = DGModel(
        balance_law,
        model.grid.numerical,
        numerical_flux_first_order,
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    return rhs
end



__________
abstract type TurbulenceClosure end

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

dissipation = ConstantViscosity{FT}(μ = 0, ν = 0, κ = 0)

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = dissipation,
    coriolis = nothing,
    buoyancy = nothing,
)

___

timestepper = TimeStepper(method = method, timestep = Δt)

simulation = Simulation(
    model = model,
    initial_conditions = initial_conditions,
    timestepper = timestepper,
    callbacks = callbacks,
    simulation_time = (start_time, end_time),
)

_______
=#








# timestepper = TimeStepper(method = method, timestep = Δt)

# simulation = Simulation(
#     model = model,
#     initial_conditions = initial_conditions,
#     timestepper = timestepper,
#     callbacks = callbacks,
#     simulation_time = (start_time, end_time),
# )




# solve
# solve!(Q_DG, solver; timeend = dt)

# viz 



#=
function evolve!(simulation, spatialmodel)
    Q = simulation.state

    # actually apply initial conditions
    for s in keys(simulation.initial_conditions)
        x, y, z = coordinates(simulation)
        p = spatialmodel.parameters
        ic = simulation.initial_conditions[s]
        ϕ = getproperty(Q, s)
        set_ic!(ϕ, ic, x, y, z, p)
    end

    Ns = polynomialorders(spatialmodel)

    if haskey(spatialmodel.numerics, :overintegration)
        Nover = spatialmodel.numerics.overintegration
    else
        Nover = (0, 0, 0)
    end
    dg = simulation.model

    if sum(Nover) > 0
        cutoff = CutoffFilter(dg.grid, Ns .- (Nover .- 1))
        num_state_prognostic = number_states(dg.balance_law, Prognostic())
        Filters.apply!(Q, 1:num_state_prognostic, dg.grid, cutoff)
    end

    function custom_tendency(tendency, x...; kw...)
        dg(tendency, x...; kw...)
        if sum(Nover) > 0
            cutoff = CutoffFilter(dg.grid, Ns .- (Nover .- 1))
            num_state_prognostic = number_states(dg.balance_law, Prognostic())
            Filters.apply!(tendency, 1:num_state_prognostic, dg.grid, cutoff)
        end
    end

    Δt = simulation.timestepper.timestep
    timestepper = simulation.timestepper.method

    odesolver = timestepper(
        custom_tendency,
        Q,
        dt = Δt,
        t0 = simulation.simulation_time[1],
    )

    cbvector = [nothing] # create_callbacks(simulation)

    if cbvector == [nothing]
        solve!(Q, odesolver; timeend = simulation.simulation_time[2])
    else
        solve!(
            Q,
            odesolver;
            timeend = simulation.simulation_time[2],
            callbacks = cbvector,
        )
    end
    return Q
end
=#