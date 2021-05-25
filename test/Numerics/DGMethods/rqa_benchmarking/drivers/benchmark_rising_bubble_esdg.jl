#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    R_d  = get_planet_parameter(:R_d),
    pₒ   = get_planet_parameter(:MSLP),
    κ    = get_planet_parameter(:kappa_d),
    g    = get_planet_parameter(:grav),
    cp_d = get_planet_parameter(:cp_d),
    cv_d = get_planet_parameter(:cv_d),
    γ    = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    T_0  = 0.0,
    xc   = 5000,
    yc   = 1000,
    zc   = 2000,
    rc   = 2000,
    xmax = 10000,
    ymax = 500,
    zmax = 10000,
    θₐ   = 2.0,
    cₛ   = 340,
)


########
# Set up domain
########
Ωˣ = IntervalDomain(min = 0, max = parameters.xmax, periodic = true)
Ωʸ = IntervalDomain(min = 0, max = parameters.ymax, periodic = true)
Ωᶻ = IntervalDomain(min = 0, max = parameters.zmax, periodic = false)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = (10, 1, 10),
    polynomial_order = (4, 4, 4),
    overintegration_order = (0, 0, 0),
)

########
# Set up model physics
########
# eos = DryIdealGas{Float64}(R = parameters.R_d, pₒ = parameters.pₒ, γ = 1 / (1 - parameters.κ))
# eos     = TotalEnergy(γ = 1 / (1 - parameters.κ))
eos = IdealGas{(:ρ, :ρu, :ρe)}()
physics = Physics(
    orientation = FlatOrientation(),
    ref_state   = NoReferenceState(),
    eos         = eos,
    lhs         = (
        NonlinearAdvection{(:ρ, :ρu, :ρe)}(),
        PressureDivergence{(:ρ, :ρu, :ρe)}(),
    ),
    sources     = (
    ),
    parameters  = parameters,
)

########
# Set up inital condition
########
r(p, x, z)          = sqrt((x - p.xc)^2 + (z - p.zc)^2)
Δθ(p, x, y, z)      = (r(p, x, z) < p.rc) ? ( p.θₐ * (1.0 - r(p, x, z) / p.rc) ) : 0
θ₀(p, x, y, z)      = 300.0 + Δθ(p, x, y, z)
π_exner(p, x, y, z) = 1.0 - p.g / (p.cp_d * θ₀(p, x, y, z) ) * z  

ρ₀(p, x, y, z)  = p.pₒ / (p.R_d * θ₀(p, x, y, z)) * (π_exner(p, x, y, z))^(p.cv_d / p.R_d)
ρu⃗₀(p, x, y, z) = @SVector [0, 0, 0]

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = p.cv_d * θ₀(p, x, y, z) * π_exner(p, x, y, z)
e_kin(p, x, y, z) = 0.0
ρe(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))
ρq(p, x, y, z) = 0

# ########
# # Set up boundary conditions
# ########
# bcs = (
#     bottom = (ρu = Impenetrable(FreeSlip()), ρe = Insulating()),
#     top =    (ρu = Impenetrable(FreeSlip()), ρe = Insulating()),
# )

########
# Set up model
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (0,0,1,1,2,3),
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρe = ρe, ρq = ρq),
    numerics = (
        flux = RoeNumericalFlux(),
        # flux = RusanovNumericalFlux(),
    ),
    parameters = parameters,
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time  = 0
end_time    = 4000.0
callbacks   = (
    Info(),
    StateCheck(10),
    VTKState(iteration = Int(floor(10.0/Δt)), filepath = "./out_esdg/"),
)


########
# Set up simulation
########
simulation = Simulation(
    model;
    grid        = grid,
    timestepper = (method = SSPRK22Heuns, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
)

########
# Run the simulation
########
initialize!(simulation)
evolve!(simulation)

nothing