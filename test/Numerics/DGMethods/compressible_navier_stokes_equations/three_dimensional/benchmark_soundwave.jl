#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    L   = 30000, # domain length
    k   = 0.5,   # sinusoidal wavenumber
    ϵ   = 1e-3,  # perturbation amplitude
    ρₒ  = 1.0,   # reference density
    γ   = 7/5,
    R_d = 287.0,
    pₒ  = 1e5,
    Tₒ  = 290,
    wₒ  = 30.0,  # background velocity
    n   = 2,     # polynomial order
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
physics = FluidPhysics(;
    orientation = FlatOrientation(),
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0, κ = 0),
    eos         = DryIdealGas(γ = parameters.γ, R = parameters.R_d, pₒ = parameters.pₒ),
)

########
# Define initial conditions
########
pert(p, x, y, z) = sin(2π * p.k * x / p.L) * sin(2π * p.k * y / p.L) * sin(2π * p.k * z / p.L)

ρ₀(p, x, y, z)   = p.ρₒ + p.ϵ * pert(p, x, y, z)
ρu₀(p, x...)     = 0
ρv₀(p, x...)     = 0
ρw₀(p, x...)     = parameters.wₒ
ρθ₀(p, x...)     = ρ₀(p, x...) * parameters.Tₒ

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
    initial_conditions  = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀),
    timestepper         = timestepper,
    callbacks           = callbacks,
    time                = (; start = start_time, finish = end_time),
)

########
# Run the model
########
evolve!(simulation, model)