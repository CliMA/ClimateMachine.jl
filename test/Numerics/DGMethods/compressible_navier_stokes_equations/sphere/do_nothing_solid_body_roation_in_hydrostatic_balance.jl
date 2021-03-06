#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Setup physical and numerical domains
########
Ω =  AtmosDomain(radius = 6e6, height = 1e5)
grid = DiscretizedDomain(
    Ω;
    elements = (vertical = 4, horizontal = 4),
    polynomial_order = (vertical = 1, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1, # reference density
    cₛ = 100.0, # sound speed
    ν  = 1e-5,
    ∂θ = 0.98 / 1e5,
    α = 2e-4,
    g = 10.0,
    power = 1,
)

########
# Define timestepping parameters
########
Δtᴬ = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
Δtᴰ = min_node_distance(grid.numerical)^2 / parameters.ν * 0.25
Δt = minimum([Δtᴬ, Δtᴰ])
start_time = 0
end_time = 86400 * 0.5

method = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks = (Info(), StateCheck(10) )# , VTKState(iteration = 2880, filepath = "."))

physics = FluidPhysics(;
    orientation = SphericalOrientation(),
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = parameters.ν, κ = 0.0),
    coriolis = ThinShellCoriolis{Float64}(Ω =  2π/86400),
    gravity = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
    eos = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ)
)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(NoSlip()),
    top = Impenetrable(NoSlip()),
)
ρθ_bcs =
    (bottom = Insulating(), top = Insulating())
BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Define initial conditions
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0

ρ₀(p, λ, ϕ, r)    = (1 -  p.∂θ * (r - 6e6)^p.power/p.power * 1e5^(1-p.power)) * p.ρₒ
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρθ₀(p, λ, ϕ, r) = -ρ₀(p, λ, ϕ, r) * p.∂θ * (r - 6e6)^(p.power-1)* 1e5^(1-p.power) * (p.cₛ)^2 / (p.α * p.g)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (   ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρθ = ρθ₀ᶜᵃʳᵗ)

########
# Create the things
########
model = SpatialModel(
    balance_law = Fluid3D(),
    physics = physics,
    numerics = (flux = RoeNumericalFlux(),),
    grid = grid,
    boundary_conditions = BC,
    parameters = parameters,
)

simulation = Simulation(
    model = model,
    initial_conditions = initial_conditions,
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

########
# Run the model
########
tic = Base.time()
evolve!(simulation, model)
toc = Base.time()
time = toc - tic
println(time)