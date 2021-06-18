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
    xmax = 1500,
    ymax = 1500,
    zmax = 3000,
    qg = 22.45e-3,
    P_sfc = 1.015e5,
    θ_liq_sfc = 299.1,
    T_sfc = 300.4,
    z₁ = 520,
    z₂ = 1480,
    z₃ = 2000,
    z₄ = 3000,
    zᵥ = 700,
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
eos = MoistIdealGas()
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

function θ₀(p,x,y,z)
    θ_liq = -0
    if -0 <= z <= p.z₁
        # Well mixed layer
        θ_liq = 298.7
    elseif z > p.z₁ && z <= p.z₂
        # Conditionally unstable layer
        θ_liq = 298.7 + (z - p.z₁) * (302.4 - 298.7) / (p.z₂ - p.z₁)
    elseif z > p.z₂&& z <= p.z₃
        # Absolutely stable inversion
        θ_liq = 302.4 + (z - p.z₂) * (308.2 - 302.4) / (p.z₃ - p.z₂)
    else
        θ_liq = 308.2 + (z - p.z₃) * (311.85 - 308.2) / (p.z₄ - p.z₃)
    end
    return θ_liq 
end

###
### Velocity Piecewise Functions
###
function u(p,x,y,z)
   u₀ = -0
   if z <= p.zᵥ
       u₀ = -8.75
   else
       u₀ = -8.75 + (z - p.zᵥ) * (-4.61 + 8.75) / (p.z₄ - p.zᵥ)
   end
   return u₀ 
end
v(p,x,y,z) = -0
w(p,x,y,z) = -0

###
### Moisture Piecewise Functions
###
function q(p,x,y,z)
    q_tot = -0
    if -0 <= z <= p.z₁
        # Well mixed layer
        q_tot = 17.0 + (z / p.z₁) * (16.3 - 17.0)
    elseif z > p.z₁ && z <= p.z₂
        # Conditionally unstable layer
        q_tot = 16.3 + (z - p.z₁) * (10.7 - 16.3) / (p.z₂ - p.z₁)
    elseif z > p.z₂ && z <= p.z₃
        # Absolutely stable inversion
        q_tot = 10.7 + (z - p.z₂) * (4.2 - 10.7) / (p.z₃ - p.z₂)
    else
        q_tot = 4.2 + (z - p.z₃) * (3.0 - 4.2) / (p.z₄ - p.z₃)
    end
    return q_tot / 1000
end

###
### Pressure Function
###
function pres(p,x,y,z)
    q_pt_sfc = PhasePartition(p.qg)
    return p.P_sfc * exp(-z/(gas_constant_air(param_set, q_pt_sfc) * p.T_sfc / p.g))
end

TS(p,x,y,z) = PhaseEquil_pθq(param_set, pres(p,x,y,z), θ₀(p,x,y,z), q(p,x,y,z))
T(p,x,y,z) = air_temperature(TS(p,x,y,z))
ρ₀(p,x,y,z) = air_density(TS(p,x,y,z))

u⃗₀(p, x, y, z) = @SVector [u(p,x,y,z), 0, 0]
ρu⃗₀(p, x, y, z) = ρ₀(p,x,y,z) * u⃗₀(p,x,y,z)
e_pot(p, x, y, z) = p.g * z
e_kin(p, x, y, z) = u(p,x,y,z)^2/2
e_tot(p, x, y, z) = total_energy(e_kin(p,x,y,z), e_pot(p,x,y,z), TS(p,x,y,z))

ρe(p, x, y, z) = ρ₀(p, x, y, z) * e_tot(p,x,y,z)
ρq(p, x, y, z) = ρ₀(p, x, y, z) * q(p, x, y, z)

########
# Set up model
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (0,0,1,1,DefaultBC(),DefaultBC()),
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρe = ρe, ρq = ρq),
    numerics = (
        flux = LMARSNumericalFlux(),
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
    VTKState(iteration = Int(floor(10.0/Δt)), filepath = "./out_esdg/bomex/"),
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
