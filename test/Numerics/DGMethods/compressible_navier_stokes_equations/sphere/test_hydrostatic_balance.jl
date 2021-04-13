#!/usr/bin/env julia --project
include("../shared_source/boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

#! format: off
refVals = (
[
 [ "state",     "ρ",   1.99999999999600046319e-02,  1.00000000000004396483e+00,  5.10000000000001008083e-01,  2.91613390275248574035e-01 ],
 [ "state", "ρu[1]",  -3.13311246519756673238e-11,  3.34851852426747048856e-11,  1.50582245544617117858e-13,  4.32064975692569532007e-12 ],
 [ "state", "ρu[2]",  -3.98988708940419829624e-11,  4.12795751448340701883e-11, -1.66842288197274268178e-14,  4.34989190229689036913e-12 ],
 [ "state", "ρu[3]",  -4.16490644835465690341e-11,  4.18355107496431387673e-11,  5.18313726923951858909e-14,  4.62898170296138141882e-12 ],
 [ "state",    "ρθ",  -4.90000000000050732751e+01, -9.79999999998041548821e-01, -2.49900000000000446221e+01,  1.42890561234872102148e+01 ],
],
[
 [ "state",     "ρ",    12,    12,    12,    12 ],
 [ "state", "ρu[1]",     0,     0,     0,     0 ],
 [ "state", "ρu[2]",     0,     0,     0,     0 ],
 [ "state", "ρu[3]",     0,     0,     0,     0 ],
 [ "state",    "ρθ",    12,    12,    12,    12 ],
],
)
#! format: on

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    ########
    # Setup physical and numerical domains
    ########
    Ω = AtmosDomain(radius = 6e6, height = 1e5)
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
        ν = 1e-5,
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
    callbacks = (Info(), StateCheck(10))# , VTKState(iteration = 2880, filepath = "."))

    physics = FluidPhysics(;
        orientation = SphericalOrientation(),
        advection = NonLinearAdvectionTerm(),
        dissipation = ConstantViscosity{Float64}(
            μ = 0,
            ν = parameters.ν,
            κ = 0.0,
        ),
        coriolis = SphereCoriolis{Float64}(Ω = 2π / 86400),
        buoyancy = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
    )

    ########
    # Define boundary conditions (west east are the ones that are enforced for a sphere)
    ########
    ρu_bcs = (bottom = Impenetrable(NoSlip()), top = Impenetrable(NoSlip()))
    ρθ_bcs = (bottom = Insulating(), top = Insulating())
    BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

    ########
    # Define initial conditions
    ########
    # Earth Spherical Representation
    # longitude: λ ∈ [-π, π), λ = 0 is the Greenwich meridian
    # latitude:  ϕ ∈ [-π/2, π/2], ϕ = 0 is the equator
    # radius:    r ∈ [Rₑ - hᵐⁱⁿ, Rₑ + hᵐᵃˣ], Rₑ = Radius of sphere; hᵐⁱⁿ, hᵐᵃˣ ≥ 0

    ρ₀(p, λ, ϕ, r) =
        (1 - p.∂θ * (r - 6e6)^p.power / p.power * 1e5^(1 - p.power)) * p.ρₒ
    ρuʳᵃᵈ(p, λ, ϕ, r) = 0.0
    ρuˡᵃᵗ(p, λ, ϕ, r) = 0.0
    ρuˡᵒⁿ(p, λ, ϕ, r) = 0.0
    ρθ₀(p, λ, ϕ, r) =
        -ρ₀(p, λ, ϕ, r) *
        p.∂θ *
        (r - 6e6)^(p.power - 1) *
        1e5^(1 - p.power) *
        (p.cₛ)^2 / (p.α * p.g)

    # Cartesian Representation (boiler plate really)
    ρ₀ᶜᵃʳᵗ(p, x...) = ρ₀(p, lon(x...), lat(x...), rad(x...))
    ρu⃗₀ᶜᵃʳᵗ(p, x...) = (
        ρuʳᵃᵈ(p, lon(x...), lat(x...), rad(x...)) * r̂(x...) +
        ρuˡᵃᵗ(p, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...) +
        ρuˡᵒⁿ(p, lon(x...), lat(x...), rad(x...)) * λ̂(x...)
    )
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
    evolve!(simulation, model, refDat = refVals)
    toc = Base.time()
    time = toc - tic
    println(time)

end
