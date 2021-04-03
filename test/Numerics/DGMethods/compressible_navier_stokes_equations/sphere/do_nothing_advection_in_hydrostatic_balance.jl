#!/usr/bin/env julia --project
include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")
include("../shared_source/gradient.jl")

ClimateMachine.init()

########
# Define physical parameters and parameterizations
########
parameters = (
    a  = 6e6,
    H  = 10e3,
    g  = 9.8,
    κ  = 2/7,
    Tₒ = 300,
    R  = 287, 
    pₒ = 1e5,
    uₒ = 20,
)

########
# Setup physical and numerical domains
########
domain =  AtmosDomain(radius = parameters.a, height = parameters.H)
grid = DiscretizedDomain(
    domain;
    elements              = (vertical = 5, horizontal = 10),
    polynomial_order      = (vertical = 4, horizontal = 4),
    overintegration_order = (vertical = 1, horizontal = 1),
)

########
# Define timestepping parameters
########
Δt          = min_node_distance(grid.numerical) / 300.0 * 0.25
start_time  = 0
end_time    = 3600 
method      = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks   = (
  Info(), 
  StateCheck(100),
  VTKState(iteration = 500, filepath = "./out/"),
)

########
# Define physics
########
physics = FluidPhysics(;
    orientation = SphericalOrientation(),
    advection   = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0.0, ν = 0.0, κ = 0.0),
    coriolis    = nothing,
    gravity     = ThinShellGravity{Float64}(g = parameters.g),
    eos         = DryIdealGas{Float64}(R = parameters.R, pₒ = parameters.pₒ, γ = 1 / (1 - parameters.κ)),
)

########
# Define Numerics
########
numerics = (flux = RoeNumericalFlux(), staggering = true)

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
F1(𝒫,r)     = r - 𝒫.a
F2(𝒫,r)     = (r - 𝒫.a)/𝒫.a + (r - 𝒫.a)^2/(2*𝒫.a^2)
expo(𝒫,ϕ,r) = 𝒫.uₒ^2/(𝒫.R*𝒫.Tₒ)*(F2(𝒫,r)*cos(ϕ)^2-sin(ϕ)^2/2)-𝒫.g*F1(𝒫,r)/(𝒫.R*𝒫.Tₒ)

dudz(𝒫,r)    = 1 + (r - 𝒫.a) / 𝒫.a 
p(𝒫,λ,ϕ,r)   = 𝒫.pₒ * exp(expo(𝒫,ϕ,r)) 
ρ₀(𝒫,λ,ϕ,r)  = p(𝒫,λ,ϕ,r) / 𝒫.R / 𝒫.Tₒ
ρθ₀(𝒫,λ,ϕ,r) = 𝒫.pₒ / 𝒫.R * (p(𝒫,λ,ϕ,r) / 𝒫.pₒ)^(1 - 𝒫.κ)

uʳᵃᵈ(𝒫,λ,ϕ,r) = 0.0
uˡᵃᵗ(𝒫,λ,ϕ,r) = 0.0
uˡᵒⁿ(𝒫,λ,ϕ,r) = 𝒫.uₒ * dudz(𝒫,r) * cos(ϕ)

ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)

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
# Fix up
#######
#
#Q = simulation.state
#
#dg = simulation.model
#Ns = polynomialorders(model)
#
#if haskey(model.grid.resolution, :overintegration_order)
#    Nover = convention(model.grid.resolution.overintegration_order, Val(ndims(model.grid.domain)))
#else
#    Nover = (0, 0, 0)
#end
#
## only works if Nover > 0
#overintegration_filter!(Q, dg, Ns, Nover)
#
#x,y,z = coordinates(grid)
#r = sqrt.(x .^2 .+ y .^2 .+ z .^2)
#∇  =  Nabla(grid)
#∇r =  ∇(r)
#ρᴮ = simulation.state.ρ
#p = ρᴮ[:,1,:] .* parameters.R * parameters.Tₒ 
#∇p = ∇(p)
#tmp = ∇p ./ ∇r
#norm(tmp[:,:,1] - tmp[:,:,2]) / norm(tmp[:,:,1]) 
#norm(tmp[:,:,2] - tmp[:,:,3]) / norm(tmp[:,:,1])
#norm(tmp[:,:,3] - tmp[:,:,1]) / norm(tmp[:,:,1])
#ρᴬ = -tmp[:,:,1] / parameters.g
#maximum(abs.(ρᴬ - ρᴮ[:,1,:]))
## simulation.state.ρ[:,1,:] .= ρᴬ
## simulation.state.ρθ[:,1,:] .=
###

########
# Run the model
########
tic = Base.time()
evolve!(simulation, model)
toc = Base.time()
time = toc - tic
println(time)
