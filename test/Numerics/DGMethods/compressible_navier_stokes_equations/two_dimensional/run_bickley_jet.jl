#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("TwoDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()
##
########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
Ωʸ = IntervalDomain(-2π, 2π, periodic = false)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ;
    elements = 8,
    polynomial_order = 3,
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
end_time = 200.0
Δt = 0.02
method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10))

########
# Define physical parameters and parameterizations
########
parameters = (
    ϵ = 0.1,  # perturbation size for initial condition
    l = 0.5, # Gaussian width
    k = 0.5, # Sinusoidal wavenumber
    ρₒ = 1.0, # reference density
    c = 2,
    g = 10,
)

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0, κ = 0),
    coriolis = nothing,
    buoyancy = nothing,
)

########
# Define boundary conditions
########
ρu_bcs = (south = Impenetrable(FreeSlip()), north = Impenetrable(FreeSlip()))
ρθ_bcs = (south = Insulating(), north = Insulating())
BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Define initial conditions
########

# The Bickley jet
U₀(p, x, y, z) = cosh(y)^(-2)

# Slightly off-center vortical perturbations
Ψ₀(p, x, y, z) =
    exp(-(y + p.l / 10)^2 / (2 * (p.l^2))) * cos(p.k * x) * cos(p.k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂ʸ, +∂ˣ) ψ̃
u₀(p, x, y, z) = Ψ₀(p, x, y, z) * (p.k * tan(p.k * y) + y / (p.l^2))
v₀(p, x, y, z) = -Ψ₀(p, x, y, z) * p.k * tan(p.k * x)
θ₀(p, x, y, z) = sin(p.k * y)

ρ₀(p, x, y, z) = p.ρₒ
ρu₀(p, x...) = ρ₀(p, x...) * (p.ϵ * u₀(p, x...) + U₀(p, x...))
ρv₀(p, x...) = ρ₀(p, x...) * p.ϵ * v₀(p, x...)
ρθ₀(p, x...) = ρ₀(p, x...) * θ₀(p, x...)

ρu⃗₀(p, x...) = @SVector [ρu₀(p, x...), ρv₀(p, x...)]
initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀)

########
# Create the things
########

model = SpatialModel(
    balance_law = Fluid2D(),
    physics = physics,
    numerics = (flux = RoeNumericalFlux(),),
    grid = grid,
    boundary_conditions = BC,
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

##
visualize(simulation, model, statenames = ["ρ", "ρu", "ρv", "ρθ"], resolution = (32,32))