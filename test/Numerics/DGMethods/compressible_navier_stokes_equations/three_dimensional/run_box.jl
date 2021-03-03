#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
Ωʸ = IntervalDomain(-2π, 2π, periodic = true)
Ωᶻ = IntervalDomain(-2π, 2π, periodic = false)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 8,
    polynomial_order = 1,
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
end_time = 200.0
Δt = 0.05
method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10))

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1, # reference density
    cₛ = sqrt(10), # sound speed
)

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 1e-2, κ = 1e-2),
    coriolis = nothing,
    buoyancy = Buoyancy{Float64}(α = 2e-4, g = 10),
)

########
# Define boundary conditions
########
ρu_bcs = (
    bottom = Impenetrable(NoSlip()),
    top = Impenetrable(MomentumFlux(
        (state, aux, t) -> (@SVector [0.01 / state.ρ, -0, -0]),
    )),
)
ρθ_bcs =
    (bottom = Insulating(), top = TemperatureFlux((state, aux, t) -> (0.1)))
BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Define initial conditions
########

ρ₀(p, x, y, z) = p.ρₒ
ρu₀(p, x...) = ρ₀(p, x...) * -0
ρv₀(p, x...) = ρ₀(p, x...) * -0
ρw₀(p, x...) = ρ₀(p, x...) * -0
ρθ₀(p, x...) = ρ₀(p, x...) * 5

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
