#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    a   = 6.371e6,     # [m] planet radius
    Ω   = 7.292e-5,    # [s⁻¹] planet rotation rate
    H   = 10e3,        # [m] sphere shell height
    g   = 9.81,        # [ms⁻²] gravitational acceleration
    R   = 287.0,       # universal gas constant
    pₒ  = 1e5,         # [Pa] reference and surface pressure
    Tₒ  = 288,         # [K] reference temperature
    κ   = 1/2, #2/7,   # ratio of ideal gas consant and heat capacity
    Γ   = 0.0065,      # [Km⁻¹] lapse rate
    ω   = 0.0,         # [s⁻¹] 
    K   = 1.962e-6,    # [s⁻¹] 
    n   = 4,           # wavenumber mode of initial profile
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 10, horizontal = 4),
    polynomial_order = (vertical = 1, horizontal = 1),
    overintegration_order = (vertical = 2, horizontal = 2),
)

########
# Define timestepping parameters
########
speed       = (parameters.n * (3 + parameters.n ) * parameters.ω - 2*parameters.Ω) / 
              ((1+parameters.n) * (2+parameters.n))
numdays     = abs(45 * π / 180 / speed / 86400)

Δt          = min_node_distance(grid.numerical) / 340.0 * 0.25
start_time  = 0
end_time    = numdays * 86400
method      = LSRKEulerMethod
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (Info(), StateCheck(10), VTKState(iteration = 100, filepath = "./out/")) 

physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
    coriolis    = ThinShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = ThinShellGravity{Float64}(g = parameters.g),
    eos         = DryIdealGas{Float64}(R = parameters.R, pₒ = parameters.pₒ, γ = 1/(1-parameters.κ)),
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

########
# Define initial conditions
########

# Earth Spherical Representation
A(𝒫,ϕ)     = 𝒫.ω/2*(2*𝒫.Ω+𝒫.ω)*cos(ϕ)^2 + 1/4*𝒫.K^2*cos(ϕ)^(2*𝒫.n)*((𝒫.n+1)*cos(ϕ)^2 + (2*𝒫.n^2-𝒫.n-2)-2*𝒫.n^2*sec(ϕ)^2)
B(𝒫,ϕ)     = 2*𝒫.K*(𝒫.Ω+𝒫.ω)*((𝒫.n+1)*(𝒫.n+2))^(-1)*cos(ϕ)^(𝒫.n)*(𝒫.n^2+2*𝒫.n+2-(𝒫.n+1)^2*cos(ϕ)^2)
C(𝒫,ϕ)     = 1/4*𝒫.K^2*cos(ϕ)^(2*𝒫.n)*((𝒫.n+1)*cos(ϕ)^2-(𝒫.n+2))
Φ(𝒫,λ,ϕ)   = 𝒫.a^2*(A(𝒫,ϕ)+B(𝒫,ϕ)*sin(𝒫.n*λ)+C(𝒫,ϕ)*(2sin(𝒫.n*λ)^2-1))
T(𝒫,r)     = 𝒫.Tₒ-𝒫.Γ*(r-𝒫.a)
pₛ(𝒫,λ,ϕ)  = 𝒫.pₒ*(1-𝒫.Γ*Φ(𝒫,λ,ϕ)/𝒫.Tₒ/𝒫.g)^(𝒫.g/𝒫.Γ/𝒫.R)
pₜ(𝒫)      = 𝒫.pₒ*(1-𝒫.Γ*𝒫.H/𝒫.Tₒ)^(𝒫.g/𝒫.Γ/𝒫.R)
p(𝒫,λ,ϕ,r) = pₜ(𝒫)+(pₛ(𝒫,λ,ϕ)-pₜ(𝒫))*(T(𝒫,r)/𝒫.Tₒ)^(𝒫.g/𝒫.Γ/𝒫.R)

ρ(𝒫,λ,ϕ,r)      = p(𝒫,λ,ϕ,r)/𝒫.R/T(𝒫,r)
uˡᵒⁿ(𝒫,λ,ϕ,r)   = -𝒫.a*𝒫.ω*cos(ϕ) - 𝒫.a*𝒫.K*cos(ϕ)^(𝒫.n-1)*(𝒫.n*sin(ϕ)^2-cos(ϕ)^2)*sin(𝒫.n*λ) 
uˡᵃᵗ(𝒫,λ,ϕ,r)   = 𝒫.n*𝒫.K*𝒫.a*cos(ϕ)^(𝒫.n-1)*sin(ϕ)*sin(𝒫.n*λ) 
uʳᵃᵈ(𝒫,λ,ϕ,r)   = 0
θ(𝒫,λ,ϕ,r)      = T(𝒫,r)*(𝒫.pₒ/p(𝒫,λ,ϕ,r))^𝒫.κ

ρuˡᵒⁿ(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r)*uˡᵒⁿ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r)*uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuʳᵃᵈ(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r)*uʳᵃᵈ(𝒫,λ,ϕ,r)
ρθ₀(𝒫, λ, ϕ, r) = ρ(𝒫,λ,ϕ,r)*θ(𝒫,λ,ϕ,r)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(𝒫, x...) = ρθ₀(𝒫, lon(x...), lat(x...), rad(x...))

########
# Create the things
########
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
tic = Base.time()
evolve!(simulation, model)
toc = Base.time()
println(toc - tic)