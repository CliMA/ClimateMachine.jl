#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    a   = get_planet_parameter(:planet_radius),
    Ω   = get_planet_parameter(:Omega),
    g   = get_planet_parameter(:grav),
    κ   = get_planet_parameter(:kappa_d),
    R_d = get_planet_parameter(:R_d), 
    pₒ  = get_planet_parameter(:MSLP),
    γ   = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    H   = 30e3,
    Tₒ  = 290,
)

########
# Set up domain
########
domain = SphericalShell(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 8, horizontal = 3),
    polynomial_order      = (vertical = 2, horizontal = 4),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvection(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 1e5/4/4, κ = 0.0),
    coriolis    = DeepShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = DeepShellGravity{Float64}(g = parameters.g, a = parameters.a),
    eos         = DryIdealGas{(:ρ, :ρu, :ρθ)}(),
    parameters  = parameters,
)

########
# Set up inital condition
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ, Rₑ + H], Rₑ = Radius of sphere; H = height of atmosphere
profile(𝒫,r)   = exp(-(1 - 𝒫.a / r) * 𝒫.a * 𝒫.g / 𝒫.R_d / 𝒫.Tₒ)
#profile(𝒫,r)   = exp(-(r - 𝒫.a) * 𝒫.g / 𝒫.R_d / 𝒫.Tₒ)
#profile(𝒫,r)   = 1 - 𝒫.Δρ / 𝒫.H / 𝒫.ρₒ * (r - 𝒫.a)
ρ₀(𝒫,λ,ϕ,r)    = 𝒫.pₒ / 𝒫.R_d / 𝒫.Tₒ * profile(𝒫,r)
#ρ₀(𝒫,λ,ϕ,r)    = 𝒫.ρₒ * profile(𝒫,r)^𝒫.e / profile(𝒫,𝒫.a + 𝒫.H)^(𝒫.e-1) 
#p(𝒫,λ,ϕ,r)     = (1 + 𝒫.ϵ * sin(2π * (r - 𝒫.a))) * 𝒫.g * 𝒫.ρₒ * 𝒫.H / 𝒫.Δρ / (𝒫.e + 1) * ρ₀(𝒫,λ,ϕ,r) * profile(𝒫,r) 
ρθ₀(𝒫,λ,ϕ,r)   = 𝒫.pₒ / 𝒫.R_d * profile(𝒫,r)^(1 - 𝒫.κ) 
#ρθ₀(𝒫,λ,ϕ,r)   = 𝒫.pₒ / 𝒫.R_d * (p(𝒫,λ,ϕ,r) / 𝒫.pₒ)^(1 / 𝒫.γ)
ρuʳᵃᵈ(𝒫,λ,ϕ,r) = 0.0
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = 0.0
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = 0.0

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(𝒫, x...) = ρθ₀(𝒫, lon(x...), lat(x...), rad(x...))
########
# Set up boundary conditions
########
bcs = (
    bottom = (ρu = Impenetrable(NoSlip()), ρθ = Insulating()),
    top =    (ρu = Impenetrable(NoSlip()), ρθ = Insulating()),
)

########
# Set up model
########
model = ModelSetup(
    physics = physics,
    boundary_conditions = bcs,
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ),
    numerics = (flux = RoeNumericalFlux(), staggering = true),
)

########
# Set up time steppers
########
Δt          = min_node_distance(grid.numerical) / 300.0 * 0.25
start_time  = 0
end_time    = 86400
callbacks   = (
    Info(), 
    StateCheck(100), 
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