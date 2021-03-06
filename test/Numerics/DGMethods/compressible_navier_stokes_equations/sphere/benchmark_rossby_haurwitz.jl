#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1,           # reference density
    cₛ = sqrt(90),    # [ms⁻¹] sound speed
    R  = 6.371e6,     # [m] planet radius
    H  = 8e3,         # [m] sphere shell height
    ω  = 0.0,         # [s⁻¹] 
    K  = 7.848e-6,    # [s⁻¹] 
    n  = 4,           
    g  = 90.0,        # [ms⁻²] gravitational accelleration
    Ω  = 7.292115e-5, # [s⁻¹] planet rotation rate
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.R, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 6),
    polynomial_order = (vertical = 0, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
speed       = (parameters.n * (3 + parameters.n ) * parameters.ω - 2*parameters.Ω) / 
              ((1+parameters.n) * (2+parameters.n))
numdays     = abs(45 * π / 180 / speed / 86400)
grav_wave_c = sqrt(parameters.g * parameters.H)

Δt          = min_node_distance(grid.numerical) / grav_wave_c * 0.25
start_time  = 0
end_time    = Δt #numdays * 86400
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (Info(), StateCheck(10), VTKState(iteration = 250, filepath = "./out/")) 

physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
    coriolis    = ThinShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = nothing,
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(FreeSlip()),
    top    = Impenetrable(FreeSlip()),
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
A(p,ϕ) = p.ω/2*(2*p.Ω+p.ω)*cos(ϕ)^2 + 1/4*p.K^2*cos(ϕ)^(2*p.n)*((p.n+1)*cos(ϕ)^2 + (2*p.n^2-p.n-2)-2*p.n^2*sec(ϕ)^2)
B(p,ϕ) = 2*p.K*(p.Ω+p.ω)*((p.n+1)*(p.n+2))^(-1)*cos(ϕ)^(p.n)*(p.n^2+2*p.n+2-(p.n+1)^2*cos(ϕ)^2)
C(p,ϕ) = 1/4*p.K^2*cos(ϕ)^(2*p.n)*((p.n+1)*cos(ϕ)^2-(p.n+2))

ρ(p,λ,ϕ,r)      = p.H + p.R^2/p.g*(A(p,ϕ) + B(p,ϕ)*cos(p.n*λ) + C(p,ϕ)*cos(2*p.n*λ))
uˡᵒⁿ(p,λ,ϕ,r)   = p.R*p.ω*cos(ϕ) + p.R*p.K*cos(ϕ)^(p.n-1)*(p.n*sin(ϕ)^2-cos(ϕ)^2)*cos(p.n*λ) 
uˡᵃᵗ(p,λ,ϕ,r)   = -p.n*p.K*p.R*cos(ϕ)^(p.n-1)*sin(ϕ)*sin(p.n*λ) 
uʳᵃᵈ(p,λ,ϕ,r)   = 0

ρuˡᵒⁿ(p,λ,ϕ,r)  = ρ(p,λ,ϕ,r) * uˡᵒⁿ(p,λ,ϕ,r)
ρuˡᵃᵗ(p,λ,ϕ,r)  = ρ(p,λ,ϕ,r) * uˡᵃᵗ(p,λ,ϕ,r)
ρuʳᵃᵈ(p,λ,ϕ,r)  = ρ(p,λ,ϕ,r) * uʳᵃᵈ(p,λ,ϕ,r)

ρθ₀(p, λ, ϕ, r) = sin(ϕ)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...)  = ρ(p, lon(x...), lat(x...), rad(x...))
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
println(toc - tic)