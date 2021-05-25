#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")

########
# Set up parameters
########
parameters = (
    a    = get_planet_parameter(:planet_radius),
    g    = get_planet_parameter(:grav),
    κ    = get_planet_parameter(:kappa_d),
    R_d  = get_planet_parameter(:R_d), 
    pₒ   = get_planet_parameter(:MSLP),
    γ    = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    H    = 10e3,
    Tₒ   = 300,
    uₒ   = 20,
)

########
# Set up domain
########
domain = SphericalShell(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 5, horizontal = 10),
    polynomial_order      = (vertical = 4, horizontal = 4),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvection(),
    gravity     = ThinShellGravity{Float64}(g = parameters.g),
    eos         = DryIdealGas{(:ρ, :ρu, :ρθ)}(),
    parameters  = parameters,
)

########
# Set up inital condition
########
F1(𝒫,r)     = r - 𝒫.a
F2(𝒫,r)     = (r - 𝒫.a)/𝒫.a + (r - 𝒫.a)^2/(2*𝒫.a^2)
expo(𝒫,ϕ,r) = 𝒫.uₒ^2/(𝒫.R_d*𝒫.Tₒ)*(F2(𝒫,r)*cos(ϕ)^2-sin(ϕ)^2/2)-𝒫.g*F1(𝒫,r)/(𝒫.R_d*𝒫.Tₒ)

dudz(𝒫,r)    = 1 + (r - 𝒫.a) / 𝒫.a 
p(𝒫,λ,ϕ,r)   = 𝒫.pₒ * exp(expo(𝒫,ϕ,r)) 
ρ₀(𝒫,λ,ϕ,r)  = p(𝒫,λ,ϕ,r) / 𝒫.R_d / 𝒫.Tₒ
ρθ₀(𝒫,λ,ϕ,r) = 𝒫.pₒ / 𝒫.R_d * (p(𝒫,λ,ϕ,r) / 𝒫.pₒ)^(1 - 𝒫.κ)

uʳᵃᵈ(𝒫,λ,ϕ,r) = 0.0
uˡᵃᵗ(𝒫,λ,ϕ,r) = 0.0
uˡᵒⁿ(𝒫,λ,ϕ,r) = 𝒫.uₒ * dudz(𝒫,r) * cos(ϕ)

ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)

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
    bottom = (ρu = Impenetrable(FreeSlip()), ρθ = Insulating()),
    top    = (ρu = Impenetrable(FreeSlip()), ρθ = Insulating()),
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
end_time    = 3600
callbacks   = (
    Info(),
    StateCheck(10),
    VTKState(iteration = 500, filepath = "./out/"),
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