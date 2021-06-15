#!/usr/bin/env julia --project
include("../interface/utilities/boilerplate.jl")
include("../interface/numerics/timestepper_abstractions.jl")

########
# Set up parameters
########
parameters = (
    ρₒ  = 1.0,
    R   = 1.0,
    H   = 0.2,
    R_d = get_planet_parameter(:R_d),
    γ   = get_planet_parameter(:cp_d)/get_planet_parameter(:cv_d),
    g   = 0.0, 
)

########
# Set up domain
########
domain = SphericalShell(
    radius = parameters.a,
    height = parameters.H,
)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 5),
    polynomial_order = (vertical = 1, horizontal = 3),
    overintegration_order = (vertical = 0, horizontal = 0),
)

########
# Define initial conditions
########
ρ₀(p, λ, ϕ, r) = p.ρₒ
ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
ρe₀(p, λ, ϕ, r) = 1e6
ρq₀(p, λ, ϕ, r) = 0.0

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(p, x...) = (
    ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) +
    ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...) +
    ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...)
)
ρe₀ᶜᵃʳᵗ(p, x...) = ρe₀(p, lon(x...), lat(x...), rad(x...))
ρq₀ᶜᵃʳᵗ(p, x...) = ρq₀(p, lon(x...), lat(x...), rad(x...))

########
# Define boundary conditions (west east are the ones that are enforced for a sphere)
########
# smart defaults / explicit? (probably here explicit for clarity)
# Everything is a special case of Flux?
sst = ...
ρu_bcs = (bottom = FreeSlip(), top = FreeSlip()) 
ρe_bcs = (bottom = NoFlux(), top = NoFlux())

ρq_bcs = (
    bottom = BoundaryFlux(
        flux = (p, state, aux, t) -> p.Q,
        params = (Q = 1e-3,), # positive means removing heat from system
    ),
    top = BoundaryFlux(
        flux = (p, state, aux, t) -> p.Q,
        params = (Q = 1e-3,), # positive means removing heat from system
    ),
)
BC = (ρq = ρq_bcs, ρu = ρu_bcs, ρe = ρe_bcs)

BC = MoistHeldSuarez(parameters = )
BC = Default()

########
# Set up model physics
########
physics = Physics(
    orientation = SphericalOrientation(),
    eos         = DryIdealGas{(:ρ, :ρu, :ρe)}(),
    lhs         = (),
    sources     = (),
    parameters = parameters,
)

########
# Set up model
########
model = DryAtmosModel(
    physics = physics,
    boundary_conditions = (5, 6),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ, ρq = ρqᶜᵃʳᵗ),
    numerics = (
        flux = RoeNumericalFlux(),
    ),
)

########
# Set up time steppers (could be done automatically in simulation)
########
dx = min_node_distance(grid.numerical)
cfl = 13.5 # 13 for 10 days, 7.5 for 200+ days
Δt = cfl * dx / 330.0
start_time = 0
end_time = 10 * 24 * 3600
method = IMEX() 
callbacks = (
  Info(),
  CFL(),
)

########
# Define timestepping parameters
########
Δt = 0.1 * min_node_distance(grid.numerical)^2 / physics.dissipation.κ
start_time = 0
end_time = 20.0 * Δt
method = SSPRK22Heuns
timestepper = TimeStepper(method = method, timestep = Δt)
callbacks = (Info(), StateCheck(10))

########
# Create the things
########
simulation = Simulation(
    (Explicit(model), Implicit(linear_model),);
    grid = grid,
    timestepper = (method = method, timestep = Δt),
    time        = (start = start_time, finish = end_time),
    callbacks   = callbacks,
);

########
# Run the model
########
tic = Base.time()

@testset "State Check" begin
    evolve!(simulation)
end

toc = Base.time()
time = toc - tic
println(time)

## Check the budget
θ̄ = weightedsum(simulation.state, 5)

@testset "Exact Geometry" begin
    # Exact (with exact geometry)
    R = grid.domain.radius
    H = grid.domain.height

    θ̄ᴱ = 0
    θ̄ᴱ -= 4π * R^2 * end_time * BC.ρq.bottom.params.Q
    θ̄ᴱ -= 4π * (R + H)^2 * end_time * BC.ρq.top.params.Q

    δθ̄ᴱ = abs(θ̄ - θ̄ᴱ) / abs(θ̄ᴱ)
    println("The relative error w.r.t. exact geometry ", δθ̄ᴱ)

    @test isapprox(0, δθ̄ᴱ; atol = 1e-9)
end

# TODO: make Approx Geom test MPI safe
#=
@testset "Approx Geometry" begin
    # Exact (with approximated geometry)
    n_e = convention(grid.resolution.elements, Val(3))
    n_ijk = convention(grid.resolution.polynomial_order, Val(3))
    n_ijk =
        n_ijk .+ convention(grid.resolution.overintegration_order, Val(3))
    n_ijk = n_ijk .+ (1, 1, 1)

    Mᴴ = reshape(
        grid.numerical.vgeo[:, grid.numerical.MHid, :],
        (n_ijk..., n_e[3], 6 * n_e[1] * n_e[2]),
    )
    Aᴮᵒᵗ = sum(Mᴴ[:, :, 1, 1, :])
    Aᵀᵒᵖ = sum(Mᴴ[:, :, end, end, :])

    θ̄ᴬ = 0
    θ̄ᴬ -= Aᴮᵒᵗ * end_time * BC.ρq.bottom.params.Q
    θ̄ᴬ -= Aᵀᵒᵖ * end_time * BC.ρq.top.params.Q

    δθ̄ᴬ = abs(θ̄ - θ̄ᴬ) / abs(θ̄ᴬ)
    println("The relative error w.r.t. approximated geometry ", δθ̄ᴬ)

    @test isapprox(0, δθ̄ᴬ; atol = 1e-15)
end
=#
