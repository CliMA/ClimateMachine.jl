#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
Ωʸ = IntervalDomain(-2π, 2π, periodic = true)
Ωᶻ = IntervalDomain(-2π, 2π, periodic = true)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 16,
    polynomial_order = 1,
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
end_time = 200.0
Δt = 0.01
method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10))

########
# Define physical parameters and parameterizations
########
parameters = (
    Uₒ = 1, # reference velocity
    ρₒ = 1, # reference density
    cₛ = sqrt(10), # sound speed
)

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 1e-3, κ = 1e-3),
    coriolis = nothing,
    buoyancy = nothing,
)

########
# Define initial conditions
########

u₀(p, x, y, z) = p.Uₒ * sin(x) * cos(y) * cos(z)
v₀(p, x, y, z) = -p.Uₒ * cos(x) * sin(y) * cos(z)
w₀(p, x, y, z) = -0
θ₀(p, x, y, z) = sin(0.5 * z)

ρ₀(p, x, y, z) = p.ρₒ
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
