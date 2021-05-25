#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    g   = get_planet_parameter(:grav),
    L   = 1,     # domain length
    k   = 0.5,   # sinusoidal wavenumber
    ϵ   = 1e-3,  # perturbation amplitude
    ρₒ  = 1.0,   # reference density
    cₛ  = 1e-2,  # sound speed
    uₒ  = 1e-3,  # background velocity
    n   = 2,     # polynomial order
)

########
# Set up domain
########
Ωˣ = IntervalDomain(min = -2*parameters.L, max = 2*parameters.L, periodic = true)
Ωʸ = IntervalDomain(min = -parameters.L, max = parameters.L, periodic = true)
Ωᶻ = IntervalDomain(min = -parameters.L, max = parameters.L, periodic = false)
grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 5,
    polynomial_order = parameters.n,
    overintegration_order = Int(ceil(3/2*parameters.n + 1/2)),
)

########
# Set up model physics
########
physics = Physics(
    orientation = FlatOrientation(),
    advection   = NonLinearAdvection(),
    eos         = BarotropicFluid{(:ρ, :ρu)}(),
    parameters  = parameters,
)

########
# Set up inital condition
########
pert(p, x, y, z) = sin(2π * p.k * x / p.L) * sin(2π * p.k * y / p.L) * sin(2π * p.k * z / p.L)
ρ₀(p, x, y, z)   = p.ρₒ + p.ϵ * pert(p, x, y, z)
ρu₀(p, x...)     = p.uₒ
ρv₀(p, x...)     = 0
ρw₀(p, x...)     = 0
ρθ₀(p, x...)     = 0
ρu⃗₀(p, x...)     = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]

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
    numerics = (flux = RoeNumericalFlux(),),
    # parameters = parameters,
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time  = 0
end_time    = 1000.0
callbacks   = (
    Info(),
    StateCheck(10),
    VTKState(iteration = Int(floor(10.0/Δt)), filepath = "./out/"),
)

########
# Set up simulation
########
simulation = Simulation(
    model;
    grid = grid,
    timestepper = (method = SSPRK22Heuns, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
)
initialize!(simulation)
evolve!(simulation)

nothing