#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1,             # reference density
    cₛ = 1e-2,          # sound speed
    R  = 1.0,           # [m]
    H  = 0.2,           # [m]
    ω  = 0.0,           # [s⁻¹]
    K  = 7.848e-6,      # [s⁻¹]
    n  = 4,             # dimensionless
    Ω  = 0.0,           # [s⁻¹] 2π/86400
    κ = 1e-3,           # diffusivity
    Qᵇ = 1e-3,          # bottom flux, positive means removing heat from system
    Qᵗ = 1e-3,          # top flux, positive means removing heat from system
    α = 2e-4,           # thermal expansion coefficent
)

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = parameters.R, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 5),
    polynomial_order = (vertical = 1+0, horizontal = 3+0),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt = min_node_distance(grid.numerical)^2 / parameters.κ * 0.1
start_time = 0
end_time = 20.0Δt
method = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks = (
    Info(), 
    StateCheck(10),
) 

physics = FluidPhysics(;
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = parameters.κ),
    coriolis    = nothing,
    gravity     = Buoyancy{Float64}(α = parameters.α, g = 0),
    eos         = BarotropicFluid{Float64}(ρₒ = parameters.ρₒ, cₛ = parameters.cₛ),
)

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
ρu_bcs = (
    bottom = Impenetrable(FreeSlip()),
    top = Impenetrable(FreeSlip()),
)
ρθ_bcs =
    (bottom = TemperatureFlux((state, aux, t) -> (parameters.Qᵇ)), 
     top = TemperatureFlux((state, aux, t) -> (parameters.Qᵗ)))
BC = 

########
# Define initial conditions
########
# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0
ρ₀(p, λ, ϕ, r)    = p.ρₒ 
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρθ₀(p, λ, ϕ, r) = 0.0

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
time = toc - tic
println(time)

## Check the budget
M = massmatrix(grid.numerical)
θ = copy(simulation.state.ρθ[:,1,:]) # density is 1
θ̅ = sum(M .* θ) 

# Exact (with exact geometry)
R = grid.domain.radius
H = grid.domain.height
θ̅ᴱ = -4π * R^2 * end_time * parameters.Qᵇ
θ̅ᴱ -= 4π * (R+H)^2 * end_time * parameters.Qᵗ

# error
println("The relative error with respect to exact geometry ", abs(θ̅ - θ̅ᴱ) / abs(θ̅ᴱ))

# Exact (with approximated geometry)
n_e = convention(grid.resolution.elements, Val(3))
n_ijk = convention(grid.resolution.polynomial_order, Val(3))
n_ijk = n_ijk .+  convention(grid.resolution.overintegration_order, Val(3))
n_ijk = n_ijk .+ (1,1,1)
Mᴴ = reshape(grid.numerical.vgeo[:, grid.numerical.MHid, :], (n_ijk..., n_e[3], 6*n_e[1]*n_e[2]))
bottomarea = sum(Mᴴ[:,:,1,1,:])
toparea = sum(Mᴴ[:,:,end,end,:])

θ̅ᴬ = -bottomarea * end_time * parameters.Qᵇ
θ̅ᴬ -= toparea  * end_time * parameters.Qᵗ
println("The relative error with respect to approximated geometry ", abs(θ̅ - θ̅ᴬ) / abs(θ̅ᴬ))