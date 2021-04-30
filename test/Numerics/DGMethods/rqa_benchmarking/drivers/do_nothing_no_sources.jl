#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    ρₒ = 1,       # reference density
    cₛ = 1e-2,    # sound speed
    a  = 1.0,     # Earth radius [m]
    H  = 0.02,    # atmos height [m]
)

########
# Set up domain
########
domain = SphericalShell(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 1, horizontal = 4),
    polynomial_order      = (vertical = 1, horizontal = 4),
    overintegration_order = 1,
)

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvection(),
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Set up inital condition
########
ρ₀(p, x, y, z) = p.ρₒ
ρu₀(p, x...)   = ρ₀(p, x...) * -0
ρv₀(p, x...)   = ρ₀(p, x...) * -0
ρw₀(p, x...)   = ρ₀(p, x...) * -0
ρθ₀(p, x...)   = ρ₀(p, x...) * 1.0

ρu⃗₀(p, x...)   = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]

########
# Set up boundary conditions
########
bcs = (
    bottom = (ρu = Impenetrable(FreeSlip()), ρθ = Insulating()),
    top    = (ρu = Impenetrable(FreeSlip()), ρθ = Insulating()),
)

########
# Set up model
########
model = ModelSetup(
    physics = physics,
    boundary_conditions = bcs,
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀),
    numerics = (grid = grid, flux = RoeNumericalFlux()),
    parameters = parameters,
)

########
# Set up time steppers
########
Δt          = 0.05
start_time  = 0
end_time    = 2.0
callbacks   = (
    Info(),
    StateCheck(10),
)

########
# Set up simulation
########
simulation = Simulation(
    model;
    timestepper = (method = SSPRK22Heuns, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
)
initialize!(simulation)
evolve!(simulation)

nothing