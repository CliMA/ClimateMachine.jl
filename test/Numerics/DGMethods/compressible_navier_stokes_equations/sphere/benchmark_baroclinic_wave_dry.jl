#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Dry Baroclinic Wave for Deep Atmosphere, Ullrich etal. (2014)
########

########
# Define physical parameters 
########
parameters = (
    a   = 6.371229e6/50,
    Ω   = 7.292e-5*50,
    g   = 9.80616,
    H   = 30000.0,
    R_d = 287.0,        
    p₀  = 1.0e5,
    k   = 3.0,
    Γ   = 0.005,
    T_E = 310.0,
    T_P = 240.0,
    b   = 2.0,
    z_t = 15e3,
    λ_c = π / 9,
    ϕ_c = 2 * π / 9,
    V_p = 1.0,
    γ   = 0.0,
    κ   = 2/7,
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 5, horizontal = 8),
    polynomial_order      = (vertical = 3, horizontal = 3),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt          = min_node_distance(grid.numerical) / 340.0 * 0.25
start_time  = 0
end_time    = 1000*Δt
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (
    Info(), 
    StateCheck(1), 
    VTKState(iteration = 1, filepath = "./out/"),
)

########
# Define parameterizations
######## 
physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 0.0, κ = 0.0),
    coriolis    = DeepShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = DeepShellGravity{Float64}(g = parameters.g, a = parameters.a),
    eos         = DryIdealGas{Float64}(R = parameters.R_d, pₒ = parameters.p₀, γ = 1/(1-parameters.κ)),
)

########
# Define initial conditions
########
# additional initial condition parameters
T_0(𝒫)  = 0.5 * (𝒫.T_E + 𝒫.T_P) 
A(𝒫)    = 1.0 / 𝒫.Γ
B(𝒫)    = (T_0(𝒫) - 𝒫.T_P) / T_0(𝒫) / 𝒫.T_P
C(𝒫)    = 0.5 * (𝒫.k + 2) * (𝒫.T_E - 𝒫.T_P) / 𝒫.T_E / 𝒫.T_P
H(𝒫)    = 𝒫.R_d * T_0(𝒫) / 𝒫.g
d_0(𝒫)  = 𝒫.a / 6

# convenience functions that only depend on height
τ_z_1(𝒫,r)   = exp(𝒫.Γ * (r - 𝒫.a) / T_0(𝒫))
τ_z_2(𝒫,r)   = 1 - 2 * ((r - 𝒫.a) / 𝒫.b / H(𝒫))^2
τ_z_3(𝒫,r)   = exp(-((r - 𝒫.a) / 𝒫.b / H(𝒫))^2)
τ_1(𝒫,r)     = 1 / T_0(𝒫) * τ_z_1(𝒫,r) + B(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_2(𝒫,r)     = C(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_int_1(𝒫,r) = A(𝒫) * (τ_z_1(𝒫,r) - 1) + B(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
τ_int_2(𝒫,r) = C(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
F_z(𝒫,r)     = (1 - 3 * ((r - 𝒫.a) / 𝒫.z_t)^2 + 2 * ((r - 𝒫.a) / 𝒫.z_t)^3) * ((r - 𝒫.a) ≤ 𝒫.z_t)

# convenience functions that only depend on longitude and latitude
d(𝒫,λ,ϕ)     = 𝒫.a * acos(sin(ϕ) * sin(𝒫.ϕ_c) + cos(ϕ) * cos(𝒫.ϕ_c) * cos(λ - 𝒫.λ_c))
c3(𝒫,λ,ϕ)    = cos(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))^3
s1(𝒫,λ,ϕ)    = sin(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))
cond(𝒫,λ,ϕ)  = (0 < d(𝒫,λ,ϕ) < d_0(𝒫)) * (d(𝒫,λ,ϕ) != 𝒫.a * π)

# base-state thermodynamic variables
I_T(𝒫,ϕ,r)   = (cos(ϕ) * r / 𝒫.a)^𝒫.k - 𝒫.k / (𝒫.k + 2) * (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 2)
T(𝒫,ϕ,r)     = (𝒫.a/r)^2 * (τ_1(𝒫,r) - τ_2(𝒫,r) * I_T(𝒫,ϕ,r))^(-1) #! First term is in question
p(𝒫,ϕ,r)     = 𝒫.p₀ * exp(-𝒫.g / 𝒫.R_d * (τ_int_1(𝒫,r) - τ_int_2(𝒫,r) * I_T(𝒫,ϕ,r)))
θ(𝒫,ϕ,r)     = T(𝒫,ϕ,r) * (𝒫.p₀ / p(𝒫,ϕ,r))^𝒫.κ

# base-state velocity variables
U(𝒫,ϕ,r)     = 𝒫.g * 𝒫.k / 𝒫.a * τ_int_2(𝒫,r) * T(𝒫,ϕ,r) * ((cos(ϕ) * r / 𝒫.a)^(𝒫.k - 1) - (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 1))
u(𝒫,ϕ,r)     = -𝒫.Ω * r * cos(ϕ) + sqrt((𝒫.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(𝒫,ϕ,r))
v(𝒫,ϕ,r)     = 0.0
w(𝒫,ϕ,r)     = 0.0

# velocity perturbations
δu(𝒫,λ,ϕ,r)  = -16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * (-sin(𝒫.ϕ_c) * cos(ϕ) + cos(𝒫.ϕ_c) * sin(ϕ) * cos(λ - 𝒫.λ_c)) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δv(𝒫,λ,ϕ,r)  = 16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * cos(𝒫.ϕ_c) * sin(λ - 𝒫.λ_c) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δw(𝒫,λ,ϕ,r)  = 0.0

# CliMA prognostic variables
ρ(𝒫,λ,ϕ,r)   = p(𝒫,ϕ,r) / 𝒫.R_d / T(𝒫,ϕ,r)
ρu(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r) * (u(𝒫,ϕ,r) + δu(𝒫,λ,ϕ,r))
ρv(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r) * (v(𝒫,ϕ,r) + δv(𝒫,λ,ϕ,r))
ρw(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r) * (w(𝒫,ϕ,r) + δw(𝒫,λ,ϕ,r))
ρθ(𝒫,λ,ϕ,r)  = ρ(𝒫,λ,ϕ,r) * θ(𝒫,ϕ,r)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (  ρw(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                    + ρv(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                    + ρu(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...)) 
ρθ₀ᶜᵃʳᵗ(𝒫, x...) = ρθ(𝒫, lon(x...), lat(x...), rad(x...))

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (bottom = Impenetrable(FreeSlip()), top = Impenetrable(FreeSlip()))
ρθ_bcs = (bottom = Insulating(), top = Insulating())

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
evolve!(simulation, model)