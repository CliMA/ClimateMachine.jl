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
    cp_v = get_planet_parameter(:cp_v),
    cp_l = get_planet_parameter(:cp_l),
    cp_i = get_planet_parameter(:cp_i),
    cv_d = get_planet_parameter(:cv_d),
    cv_v = get_planet_parameter(:cv_v),
    cv_l = get_planet_parameter(:cv_l),
    cv_i = get_planet_parameter(:cv_i),
    e_int_v0 = get_planet_parameter(:e_int_v0),
    e_int_i0 = get_planet_parameter(:e_int_i0),
    molmass_ratio = get_planet_parameter(:molmass_dryair)/get_planet_parameter(:molmass_water),
    T_0  = get_planet_parameter(:T_0), # 0.0, #
    xc   = 5000,
    yc   = 1000,
    zc   = 2000,
    rc   = 2000,
    xmax = 10000,
    ymax = 500,
    zmax = 10000,
    θₐ   = 2.0,
    cₛ   = 340,
    q₀   = 1e-3,
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
eos = MoistIdealGas{(:ρ, :ρu, :ρe)}()
physics = Physics(
    orientation = FlatOrientation(),
    ref_state   = NoReferenceState(),
    eos         = eos,
    lhs         = (
        NonlinearAdvection{(:ρ, :ρu, :ρe)}(),
        PressureDivergence(),
    ),
    sources     = (
        FluctuationGravity(),
    ),
    parameters  = parameters,
)

########
# Set up inital condition
########
r(p, x, z)          = sqrt((x - p.xc)^2 + (z - p.zc)^2)
Δθ(p, x, y, z)      = (r(p, x, z) < p.rc) ? ( p.θₐ * (1.0 - r(p, x, z) / p.rc) ) : 0
θ₀(p, x, y, z)      = 300.0 + Δθ(p, x, y, z)
Δq(p, x, y, z)      = (r(p, x, z) < p.rc) ? ( p.q₀ * (1.0 - r(p, x, z) / p.rc) ) : 0
q(p, x, y, z)       = 0.0 + Δq(p, x, y, z)
π_exner(p, x, y, z) = 1.0 - p.g / (p.cp_d * θ₀(p, x, y, z) ) * z  

ρ₀(p, x, y, z)  = p.pₒ / (p.R_d * θ₀(p, x, y, z)) * (π_exner(p, x, y, z))^(p.cv_d / p.R_d)
ρu⃗₀(p, x, y, z) = @SVector [0, 0, 0]

cv_m(p, x, y, z)  = p.cv_d + (p.cv_v - p.cv_d) * q(p, x, y, z)

e_pot(p, x, y, z) = p.g * z
e_int(p, x, y, z) = cv_m(p, x, y, z) * (θ₀(p, x, y, z) * π_exner(p, x, y, z) - p.T_0) + q(p, x, y, z) * p.e_int_v0
e_kin(p, x, y, z) = 0.0

ρe(p, x, y, z) = ρ₀(p, x, y, z) * (e_kin(p, x, y, z) + e_int(p, x, y, z) + e_pot(p, x, y, z))
ρq(p, x, y, z) = ρ₀(p, x, y, z) * q(p, x, y, z)

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
    boundary_conditions = (0,0,1,1,DefaultBC(),DefaultBC()),
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρe = ρe, ρq = ρq),
    numerics = (
        # flux = RoeNumericalFlux(),
        flux = LMARSNumericalFlux(),
        # flux = RusanovNumericalFlux(),
    ),
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time  = 0
end_time    = 6000.0
callbacks   = (
    Info(),
    StateCheck(10),
    TMARCallback(),
    VTKState(iteration = Int(floor(10.0/Δt)), filepath = "./out_esdg/moist_bubble/"),
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