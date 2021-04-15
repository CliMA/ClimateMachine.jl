using ClimateMachine
ClimateMachine.init()

include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")
include("../shared_source/gradient.jl")

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1, # reference density
    cₛ = 100.0, # sound speed
    a  = 6e6,
    H  = 1e5,
    Ω  = 2π/86400,
    α  = 2e-4,
    g  = 9.81,
    ∂θ = 0.98 / 1e5,
    power = 1,
    ef = 4.0,
)

########
# Setup physical and numerical domains
########
domain =  AtmosDomain(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 3, horizontal = 3),
    polynomial_order      = (vertical = 2, horizontal = 4),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt          = min_node_distance(grid.numerical) / parameters.cₛ * 0.25 * 4
start_time  = 0
end_time    = 86400 * 0.5 * 6  # Δt # 86400 * 0.5 * 6 * 5
method      = SSPRK22Heuns # LSRKEulerMethod # SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (Info(), StateCheck(30))

########
# Define physics
########
physics = FluidPhysics(;
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 0.0, κ = 0.0),
    coriolis    = ThinShellCoriolis{Float64}(Ω = parameters.Ω),
    gravity     = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ)
)

########
# Define Numerics
########

numerics = (flux = RoeNumericalFlux(), staggering = true)
numerics = (flux = RoesanovFlux(ω_roe = 1.0, ω_rusanov = 0.1), staggering = true)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(NoSlip()),
    top = Impenetrable(NoSlip()),
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
ρ₀(p, λ, ϕ, r)    = (1 -  p.∂θ * (r - 6e6)^p.power/p.power * 1e5^(1-p.power)) * p.ρₒ
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρθ₀(p, λ, ϕ, r) = -ρ₀(p, λ, ϕ, r) * p.∂θ * (r - 6e6)^(p.power-1)* 1e5^(1-p.power) * (p.cₛ)^2 / (p.α * p.g)

ρ₀(p, λ, ϕ, r)    = exp(-p.ef*(r - 6e6)/1e5) * p.ρₒ
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρθ₀(p, λ, ϕ, r) = ρ₀(p, λ, ϕ, r) * (-p.ef/ 1e5 * exp(-p.ef*(r - 6e6)/1e5)) * (p.cₛ)^2 / (p.α * p.g)

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (   ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρθ₀ᶜᵃʳᵗ(p, x...) = ρθ₀(p, lon(x...), lat(x...), rad(x...))

########
# Create the things
########
model = SpatialModel(
    balance_law         = Fluid3D(),
    physics             = physics,
    numerics            = numerics,
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


#######
# Numerically Construct Density
#######
x,y,z = coordinates(grid)
r = sqrt.(x .^2 .+ y .^2 .+ z .^2)
∇  =  Nabla(grid)
∇r =  ∇(r)
ρᴬ = simulation.state.ρ
p = (ρᴬ[:,1,:] .^2) .* (parameters.cₛ)^2 / parameters.ρₒ /2
∇p = ∇(p)
ρθᴮ = (x .* ∇p[:,:,1] + y .* ∇p[:,:,2] + z .* ∇p[:,:,3])
ρθᴮ = ρθᴮ ./ (x .* ∇r[:,:,1] + y .* ∇r[:,:,2] + z .* ∇r[:,:,3])
ρθᴮ = ρθᴮ ./ (parameters.α * parameters.g)
norm(simulation.state.ρθ[:,1,:] - ρθᴮ)
er1 = maximum(abs.( -∇p[:,:,1] + parameters.α * parameters.g * ρθᴮ .* ∇r[:,:,1]))
er2 = maximum(abs.( -∇p[:,:,1] + parameters.α * parameters.g * simulation.state.ρθ[:,1,:] .* ∇r[:,:,1]))
# simulation.state.ρθ[:,1,:] = ρθᴮ # set equal to numerical one

##
########
# Run the model
########
tic = Base.time()
evolve!(simulation, model)
toc = Base.time()
time = toc - tic
println(time)
