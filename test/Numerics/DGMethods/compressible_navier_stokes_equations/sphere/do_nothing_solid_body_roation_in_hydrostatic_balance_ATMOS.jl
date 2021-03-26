#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    a  = 6.37e6/125.0,
    H  = 30e3,
    Ω  = 2π/86400,
    g  = 9.81,
    R  = 287,
    pₒ = 1e5,
    Tₒ = 290,
    κ  = 2/7,
)

########
# Setup physical and numerical domains
########
domain =  AtmosDomain(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 5, horizontal = 4),
    polynomial_order      = (vertical = 3, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt          = min_node_distance(grid.numerical) / 340.0 * 0.25
start_time  = 0
end_time    = 86400 * 0.5
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (Info(), StateCheck(400))

########
# Define physics
########
physics = FluidPhysics(;
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 0.0, κ = 0.0),
    coriolis    = DeepShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = DeepShellGravity{Float64}(g = parameters.g, a = parameters.a),
    #gravity     = ThinShellGravity{Float64}(g = parameters.g),
    eos         = DryIdealGas{Float64}(R = parameters.R, pₒ = parameters.pₒ, γ = 1 / (1 - parameters.κ)),
)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(FreeSlip()),
    top = Impenetrable(FreeSlip()),
)
ρθ_bcs =
    (bottom = Insulating(), top = Insulating())
    
########
# Define initial conditions
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0
profile(𝒫,r)   = exp(-(1 - 𝒫.a / r) * 𝒫.a * 𝒫.g / 𝒫.R / 𝒫.Tₒ)
#profile(𝒫,r)   = exp(-(r - 𝒫.a) * 𝒫.g / 𝒫.R / 𝒫.Tₒ)
ρ₀(𝒫,λ,ϕ,r)    = 𝒫.pₒ / 𝒫.R / 𝒫.Tₒ * profile(𝒫,r)
ρuʳᵃᵈ(𝒫,λ,ϕ,r) = 0.0
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = 0.0
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = 0.0
ρθ₀(𝒫,λ,ϕ,r)   = 𝒫.pₒ / 𝒫.R * profile(𝒫,r)^(1 - 𝒫.κ) 

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(𝒫, x...) = ρθ₀(𝒫, lon(x...), lat(x...), rad(x...))

########
# Create the things
########
model = SpatialModel(
    balance_law         = Fluid3D(),
    physics             = physics,
    numerics            = (flux = RoeNumericalFlux(),),
    grid                = grid,
    boundary_conditions = (ρθ = ρθ_bcs, ρu = ρu_bcs),
    parameters          = parameters,
)

simulation = Simulation(
    model               = model,
    initial_conditions  = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ),
    timestepper         = timestepper,
    callbacks           = callbacks,
    time                = (; start = start_time, finish = end_time),
)

########
# Run the model
########
tic = Base.time()
evolve!(simulation, model)
toc = Base.time()
time = toc - tic
println(time)
