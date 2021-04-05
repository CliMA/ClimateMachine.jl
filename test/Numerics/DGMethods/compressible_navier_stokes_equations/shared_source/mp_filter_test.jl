include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("../sphere/sphere_helper_functions.jl")

ClimateMachine.init()

########
# Setup physical and numerical domains
########
domain = AtmosDomain(radius = 1, height = 0.2)

grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 3),
    polynomial_order = (vertical = 1+0, horizontal = 3+0),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1,             # reference density
    cₛ = 1e-2,          # sound speed
)

########
# Define timestepping parameters
########
Δt = 1
start_time = 0
end_time = Δt

method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10),) 

physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
    coriolis = nothing,
    buoyancy = Buoyancy{Float64}(α = 0, g = 0),
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
    (bottom = TemperatureFlux((state, aux, t) -> (parameters.Qᵇ)), 
     top = TemperatureFlux((state, aux, t) -> (parameters.Qᵗ)))
BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Define initial conditions
########

# Earth Spherical Representation
# longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
# latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
# radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0

ρ₀(p, λ, ϕ, r)    = p.ρₒ * exp(λ ) * exp(ϕ) * exp(r)
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

##
simulation = Simulation(
    model = model,
    initial_conditions = initial_conditions,
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

Ns = polynomialorders(model)
Nover = convention(model.grid.resolution.overintegration_order, Val(ndims(model.grid.domain)))
dgmodel = simulation.model
cutoff_order = Ns .+ 1 # yes this is confusing

## Intialize

M = Array(massmatrix(dgmodel.grid))
ρᴮ = Array(copy(simulation.state.ρ))
ρ̅ᴮ =  sum(M .* ρᴮ[:,1,:], dims = 1) ./ sum(M, dims = 1)

cutoff = MassPreservingCutoffFilter(dgmodel.grid, cutoff_order)
# cutoff = ClimateMachine.Mesh.Filters.CutoffFilter(dgmodel.grid, cutoff_order)
num_state_prognostic = number_states(dgmodel.balance_law, Prognostic())

ClimateMachine.Mesh.Filters.apply!(
    simulation.state,
    1:num_state_prognostic,
    dgmodel.grid,
    cutoff,
)

ρᴬ = Array(copy(simulation.state.ρ))
ρ̅ᴬ =  sum(M .* ρᴬ[:,1,:], dims = 1) ./ sum(M, dims = 1)

println("Mass Preserving CutOff Filter")
println("The solution changed by " , norm(ρᴬ - ρᴮ) / norm(ρᴬ) )
println("The maximum mass average per element has changed by ", maximum( abs.(ρ̅ᴮ .- ρ̅ᴬ)))

## Check Idempotency
ρᴮ = Array(copy(simulation.state.ρ))
ρ̅ᴮ =  sum(M .* ρᴮ[:,1,:], dims = 1) ./ sum(M, dims = 1)

cutoff = MassPreservingCutoffFilter(dgmodel.grid, cutoff_order)
# cutoff = ClimateMachine.Mesh.Filters.CutoffFilter(dgmodel.grid, cutoff_order)
num_state_prognostic = number_states(dgmodel.balance_law, Prognostic())

ClimateMachine.Mesh.Filters.apply!(
    simulation.state,
    1:num_state_prognostic,
    dgmodel.grid,
    cutoff,
)

ρᴬ = Array(copy(simulation.state.ρ))
ρ̅ᴬ =  sum(M .* ρᴬ[:,1,:], dims = 1) ./ sum(M, dims = 1)

println("Idempotency of Mass Preserving CutOff Filter")
println("The solution changed by " , norm(ρᴬ - ρᴮ) / norm(ρᴬ) )
println("The maximum mass average per element has changed by ", maximum( abs.(ρ̅ᴮ .- ρ̅ᴬ)))

##
simulation = Simulation(
    model = model,
    initial_conditions = initial_conditions,
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

ρᴮ = Array(copy(simulation.state.ρ))
ρ̅ᴮ =  sum(M .* ρᴮ[:,1,:], dims = 1) ./ sum(M, dims = 1)

# cutoff = MassPreservingCutoffFilter(dgmodel.grid, cutoff_order)
cutoff = ClimateMachine.Mesh.Filters.CutoffFilter(dgmodel.grid, cutoff_order)
num_state_prognostic = number_states(dgmodel.balance_law, Prognostic())

ClimateMachine.Mesh.Filters.apply!(
    simulation.state,
    1:num_state_prognostic,
    dgmodel.grid,
    cutoff,
)

ρᴬ = Array(copy(simulation.state.ρ))
ρ̅ᴬ =  sum(M .* ρᴬ[:,1,:], dims = 1) ./ sum(M, dims = 1)

println("CutOff Filter")
println("The solution changed by " , norm(ρᴬ - ρᴮ) / norm(ρᴬ) )
println("The maximum mass average per element has changed by ", maximum( abs.(ρ̅ᴮ .- ρ̅ᴬ)))
