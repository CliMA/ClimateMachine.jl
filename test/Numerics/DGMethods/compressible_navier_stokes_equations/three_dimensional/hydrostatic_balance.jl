#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
Ωʸ = IntervalDomain(-2π, 2π, periodic = true)
Ωᶻ = IntervalDomain(0, 2π, periodic = false)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 4,
    polynomial_order = 4,
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
Δt = 0.01
end_time = 1000Δt

method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10))

########
# Define physical parameters and parameterizations
########
parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    ρₒ = 1, # reference density
    cₛ = sqrt(10), # sound speed
    α = 1e-6,
    g = 10.0,
    θ = 100,
)

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0, κ = 0),
    coriolis = nothing,
    buoyancy = buoyancy = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
)

########
# Define initial conditions
########


# Vortical velocity fields (u, v, w) = (-∂ʸ, +∂ˣ, 0) Ψ₁ + (0, -∂ᶻ, +∂ʸ)Ψ₂ 
u₀(p, x, y, z) = 0.0
v₀(p, x, y, z) = 0.0
w₀(p, x, y, z) = 0.0
θ₀(p, x, y, z) = -1.0 * p.θ

ρ₀(p, x, y, z) = p.ρₒ / (p.cₛ^2) * (1 - p.θ * p.α * p.g * z  )
ρu₀(p, x...) = ρ₀(p, x...) * u₀(p, x...) 
ρv₀(p, x...) = ρ₀(p, x...) * v₀(p, x...) 
ρw₀(p, x...) = ρ₀(p, x...) * w₀(p, x...) 
ρθ₀(p, x...) = ρ₀(p, x...) * θ₀(p, x...)

ρu⃗₀(p, x...) = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]
initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀)

########
# Create the things
########

model = SpatialModel(
    balance_law = Fluid3D(),
    physics = physics,
    numerics = (flux = RoeNumericalFlux(),),
    grid = grid,
    boundary_conditions = NamedTuple(),
    parameters = parameters,
)

simulation = Simulation(
    model = model,
    initial_conditions = initial_conditions,
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

########
# Run the model
########

tic = Base.time()

evolve!(simulation, model)

toc = Base.time()
time = toc - tic
println(time)