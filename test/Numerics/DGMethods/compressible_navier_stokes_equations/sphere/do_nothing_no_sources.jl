#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1,    # reference density
    cₛ = 1e-2, # sound speed
    R  = 1.0,           # [m]
    H  = 0.02,           # [m]
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.R, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 4),
    polynomial_order = (vertical = 1, horizontal = 4),
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
end_time = 2.0
Δt = 0.05
method = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks = (
    Info(), 
    StateCheck(10)
)

physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
    coriolis    = nothing,
    gravity     = nothing,
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(FreeSlip()),
    top = Impenetrable(FreeSlip()),
)
ρθ_bcs =
    (bottom = Insulating(), top = Insulating())

########
# Define initial conditions
########
ρ₀(p, x, y, z) = p.ρₒ
ρu₀(p, x...)   = ρ₀(p, x...) * -0
ρv₀(p, x...)   = ρ₀(p, x...) * -0
ρw₀(p, x...)   = ρ₀(p, x...) * -0
ρθ₀(p, x...)   = ρ₀(p, x...) * 1.0

ρu⃗₀(p, x...)   = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]

########
# Create the things
########
model = SpatialModel(
    balance_law = Fluid3D(),
    physics = physics,
    numerics = (flux = RoeNumericalFlux(),),
    grid = grid,
    boundary_conditions = (ρθ = ρθ_bcs, ρu = ρu_bcs),
    parameters = parameters,
)

simulation = Simulation(
    model = model,
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀),
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