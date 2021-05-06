#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    ρₒ = 1,          # reference density
    cₛ = 100.0,      # sound speed
    a  = 6e6,        # Earth radius
    H  = 1e5,        # atmos height
    Ω  = 2π/86400,   # Earth angular velocity
    α  = 2e-4,       # buoyancy scaling [K⁻¹]
    g  = 9.81,       # gravity of Earth
    ∂θ = 0.98 / 1e5, 
    power = 1,       
    ef = 4.0,       
)

########
# Set up domain
########
domain = SphericalShell(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 3, horizontal = 3),
    polynomial_order      = (vertical = 2, horizontal = 4),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvection(),
    coriolis    = ThinShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Set up inital condition
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ, Rₑ+H], Rₑ = Radius of sphere; H = height of atmosphere
ρ₀(p, λ, ϕ, r)    = (1 -  p.∂θ * (r - 6e6)^p.power/p.power * 1e5^(1-p.power)) * p.ρₒ
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρθ₀(p, λ, ϕ, r) = -ρ₀(p, λ, ϕ, r) * p.∂θ * (r - 6e6)^(p.power-1)* 1e5^(1-p.power) * (p.cₛ)^2 / (p.α * p.g)

# ρ₀(p, λ, ϕ, r)    = exp(-p.ef*(r - 6e6)/1e5) * p.ρₒ
# ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
# ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
# ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
# ρθ₀(p, λ, ϕ, r) = ρ₀(p, λ, ϕ, r) * (-p.ef/ 1e5 * exp(-p.ef*(r - 6e6)/1e5)) * (p.cₛ)^2 / (p.α * p.g)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (   ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

########
# Set up boundary conditions
########
bcs = (
    bottom = (ρu = Impenetrable(NoSlip()), ρθ = Insulating()),
    top    = (ρu = Impenetrable(NoSlip()), ρθ = Insulating()),
)

########
# Set up model
########
model = ModelSetup(
    physics = physics,
    boundary_conditions = bcs,
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ),
    numerics = (
        grid = grid, 
        flux = RoesanovFlux(ω_roe = 1.0, ω_rusanov = 0.1), 
        staggering = true
    ),
    parameters = parameters,
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time  = 0
end_time    = 86400 * 0.5 * 6
callbacks   = (
    Info(),
    StateCheck(10),
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
initialize!(simulation)
evolve!(simulation)

nothing