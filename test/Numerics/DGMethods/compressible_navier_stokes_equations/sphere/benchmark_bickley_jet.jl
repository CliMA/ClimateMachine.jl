#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Define physical parameters 
########
parameters = (
    ρₒ = 1.0,    # reference density
    cₛ = 1e-2, # sound speed
    ℓᵐ = 10,   # jet thickness, (larger is thinner)
    ℓ = 20,    # perturbation thickness, (larger is thinner)
    m = 2,     # number of sign changes on equator for perturbation
    ϕᵖ = π/2 * 0.05, # of centerdness of perturbation
    ϵ = 0.3,   # perturbation amplitude
    vˢ = 5e-4, # velocity scale
    α = 2e-4,
    Ω = 1e-3,
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = 1, height = 0.2)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 8),
    polynomial_order = (vertical = 0, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt = min_node_distance(grid.numerical) / parameters.cₛ * 0.25
start_time = 0
end_time = 8*1600.0
method = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (
    Info(), 
    StateCheck(10), 
    VTKState(iteration = 160, filepath = "./out/"),
)

########
# Define parameterizations
########
physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
    coriolis = DeepShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity = Buoyancy{Float64}(α = parameters.α, g = 0),
    eos = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
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
BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Define initial conditions
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of earth; hᵐⁱⁿ, hᵐᵃˣ ≥ 0
# u =   - r⁻¹∂ψ/∂ϕ
# v =  ( r cos(ϕ) )⁻¹ ∂ψ/∂λ
#ψᵐ(λ, ϕ) = tanh(ℓᵐ * ϕ) 
#ψᵖ(λ, ϕ) = exp(-ℓ * (ϕ - ϕᵖ)^2) * cos(ϕ) * cos(2 * (ϕ - ϕᵖ)) * sin(m * λ)
uᵐ(p, λ, ϕ, r) =  p.ℓᵐ * sech(p.ℓᵐ * ϕ)^2 
vᵐ(p, λ, ϕ, r) =  0.0
hᵐ(p, λ, ϕ, r) =  0.0 

u1(p, λ, ϕ, r) =  p.ℓ * 2 * (ϕ - p.ϕᵖ)* exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u2(p, λ, ϕ, r) =    exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * sin(ϕ) * cos(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
u3(p, λ, ϕ, r) =  2*exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(ϕ) * sin(2 * (ϕ - p.ϕᵖ)) * sin(p.m * λ)
uᵖ(p, λ, ϕ, r) =  u1(p, λ, ϕ, r) + u2(p, λ, ϕ, r) + u3(p, λ, ϕ, r)
vᵖ(p, λ, ϕ, r) =  p.m * exp(-p.ℓ * (ϕ - p.ϕᵖ)^2) * cos(2 * (ϕ - p.ϕᵖ)) * cos(p.m * λ)
hᵖ(p, λ, ϕ, r) =  0.0 

ρ₀(p, λ, ϕ, r)    = p.ρₒ 
ρuʳᵃᵈ(p, λ, ϕ, r) = 0
ρuˡᵃᵗ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (p.ϵ * vᵖ(p, λ, ϕ, r))
ρuˡᵒⁿ(p, λ, ϕ, r) = p.vˢ * ρ₀(p, λ, ϕ, r) * (uᵐ(p, λ, ϕ, r) + p.ϵ * uᵖ(p, λ, ϕ, r))
ρθ₀(p, λ, ϕ, r) = ρ₀(p, λ, ϕ, r) * tanh(p.ℓᵐ * ϕ)

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
M = massmatrix(model.grid.numerical)
for i in 1:5
   println("budget before: " , sum(M .* simulation.state[:,i,:]) )
end
evolve!(simulation, model)
for i in 1:5
   println("budget after: " , sum(M .* simulation.state[:,i,:]) )
end
toc = Base.time()
time = toc - tic
println(time)