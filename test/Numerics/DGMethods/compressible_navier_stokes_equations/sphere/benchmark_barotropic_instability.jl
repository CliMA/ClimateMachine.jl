#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

using QuadGK: quadgk

ClimateMachine.init()

########
# Shallow-Water-ish Barotropic Instability, Galewsky etal. (2004)
########

########
# Define physical parameters 
########
parameters = (
    a    = 6.37122e6,
    Ω    = 7.292e-5,
    g    = 9.80616,
    cₛ   = sqrt(9.80616),
    uₘₐₓ = 80.0,
    eₙ   = exp(-4(3π/14)^-2), 
    ϕ₀   = π/7,
    ϕ₁   = 5π/14,
    ϕ₂   = π/4,
    α    = 1/3,
    β    = 1/15,                
    ρₒ   = 1.0,
    ρᵢ   = 15000.0,
    ρ̂    = 120.0,
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.a, height = 0.01*parameters.a)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 1, horizontal = 32),
    polynomial_order      = (vertical = 1, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt          = min_node_distance(grid.numerical) / (parameters.cₛ * sqrt(parameters.ρᵢ / parameters.ρₒ)) * 0.25
start_time  = 0
end_time    = 6*24*60*60
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (
    Info(), 
    StateCheck(144), 
    VTKState(iteration = 1800, filepath = "./out/"),
)

########
# Define parameterizations
########
physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 0.0, κ = 0.0),
    coriolis    = ThinShellCoriolis{Float64}(Ω = parameters.Ω),
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Define initial conditions
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of earth; hᵐⁱⁿ, hᵐᵃˣ ≥ 0
uˡᵒⁿ(p,λ,ϕ,r) = r / p.a * p.uₘₐₓ / p.eₙ * exp(1 / (ϕ - p.ϕ₀) / (ϕ - p.ϕ₁)) * (p.ϕ₀ < ϕ < p.ϕ₁)

ρᵢₙₜ(p,λ,ϕ,r)  = p.a * uˡᵒⁿ(p, λ, ϕ, r) * (2 * p.Ω * sin(ϕ) + tan(ϕ) * uˡᵒⁿ(p, λ, ϕ, r) / p.a) / p.g
ρₚ(p,λ,ϕ,r)    = p.ρ̂ * cos(ϕ) * exp(-λ^2 / p.α^2 - (p.ϕ₂ - ϕ)^2 / p.β^2)
ρ₀(p,λ,ϕ,r)    = p.ρᵢ - quadgk(φ -> ρᵢₙₜ(p, λ, φ, r), -π/2, ϕ)[1] + ρₚ(p, λ, ϕ, r)

ρuˡᵒⁿ(p,λ,ϕ,r) = ρ₀(p,λ,ϕ,r) * uˡᵒⁿ(p,λ,ϕ,r)
ρuˡᵃᵗ(p,λ,ϕ,r) = 0.0
ρuʳᵃᵈ(p,λ,ϕ,r) = 0.0

ρθ₀(p,λ,ϕ,r)   = sign(ϕ - 0.5*(p.ϕ₁ + p.ϕ₀))

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...)  = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (   ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (bottom = Impenetrable(FreeSlip()), top = Impenetrable(FreeSlip()))
ρθ_bcs = (bottom = Insulating(), top = Insulating())

#######
# Create the things
#######
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
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ),
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

########
# Run the model
########
evolve!(simulation, model)