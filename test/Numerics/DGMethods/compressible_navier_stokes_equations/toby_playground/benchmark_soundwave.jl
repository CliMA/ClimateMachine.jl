#!/usr/bin/env julia --project

include("boilerplate.jl")
include("BasicLaw.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1.0,
    cₛ = 1e-3,
    L  = 1,
    k  = 0.5,
    n  = 3,
    ϵ  = 0.01,
)

########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(-parameters.L, parameters.L, periodic = true)
Ωʸ = IntervalDomain(-parameters.L, parameters.L, periodic = true)
Ωᶻ = IntervalDomain(-2*parameters.L, 2*parameters.L, periodic = true)
grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 10,
    polynomial_order = parameters.n,
    overintegration_order = Int(ceil(3/2*parameters.n + 1/2)),
)

########
# Define timestepping parameters
########
start_time  = 0
end_time    = 1000.0
Δt          = min_node_distance(grid.numerical) / 340 * 0.25
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (
    Info(),
    StateCheck(10),
    VTKState(iteration = Int(floor(10.0/Δt)), filepath = "./out/"),
)

########
# Define physical parameters
########
physics = FluidPhysics(;)

########
# Define initial conditions
########
pert(p, x, y, z) = sin(2π * p.k * x / p.L) * sin(2π * p.k * y / p.L) * sin(2π * p.k * z / p.L)

ρ₀(p, x, y, z)   = p.ρₒ + p.ϵ * pert(p, x, y, z)
ρu₀(p, x...)     = 0
ρv₀(p, x...)     = 0
ρw₀(p, x...)     = 0

ρu⃗₀(p, x...)     = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]

########
# Create the things
########
model = SpatialModel(
    balance_law         = Fluid3D(),
    physics             = physics,
    numerics            = (flux = RoeNumericalFlux(),),
    grid                = grid,
    boundary_conditions = NamedTuple(),
    parameters          = parameters,
)

simulation = Simulation(
    model               = model,
    initial_conditions  = (ρ = ρ₀, ρu = ρu⃗₀,),
    timestepper         = timestepper,
    callbacks           = callbacks,
    time                = (; start = start_time, finish = end_time),
)

########
# Run the model
########
evolve!(simulation, model)