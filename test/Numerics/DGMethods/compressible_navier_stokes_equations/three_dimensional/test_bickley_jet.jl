#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("refvals_bickley_jet.jl")

    ########
    # Setup physical and numerical domains
    ########
    Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
    Ωʸ = IntervalDomain(-2π, 2π, periodic = true)
    Ωᶻ = IntervalDomain(-2π, 2π, periodic = true)

    first_order = DiscretizedDomain(
        Ωˣ × Ωʸ × Ωᶻ;
        elements = 32,
        polynomial_order = 1,
        overintegration_order = 1,
    )

    fourth_order = DiscretizedDomain(
        Ωˣ × Ωʸ × Ωᶻ;
        elements = 13,
        polynomial_order = 4,
        overintegration_order = 1,
    )

    grids = Dict("first_order" => first_order, "fourth_order" => fourth_order)

    ########
    # Define timestepping parameters
    ########
    start_time = 0
    end_time = 200.0
    Δt = 0.004
    method = SSPRK22Heuns

    timestepper = TimeStepper(method = method, timestep = Δt)

    callbacks = (Info(), StateCheck(10))

    ########
    # Define physical parameters and parameterizations
    ########
    parameters = (
        ϵ = 0.1,  # perturbation size for initial condition
        l = 0.5, # Gaussian width
        k = 0.5, # Sinusoidal wavenumber
        ρₒ = 1, # reference density
        cₛ = sqrt(10), # sound speed
    )

    physics = FluidPhysics(;
        advection = NonLinearAdvectionTerm(),
        dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0, κ = 0),
        coriolis = nothing,
        buoyancy = nothing,
    )

    ########
    # Define initial conditions
    ########

    # The Bickley jet
    U₀(p, x, y, z) = cosh(y)^(-2)
    V₀(p, x, y, z) = 0
    W₀(p, x, y, z) = 0

    # Slightly off-center vortical perturbations
    Ψ₁(p, x, y, z) =
        exp(-(y + p.l / 10)^2 / (2 * (p.l^2))) * cos(p.k * x) * cos(p.k * y)
    Ψ₂(p, x, y, z) =
        exp(-(z + p.l / 10)^2 / (2 * (p.l^2))) * cos(p.k * y) * cos(p.k * z)

    # Vortical velocity fields (u, v, w) = (-∂ʸ, +∂ˣ, 0) Ψ₁ + (0, -∂ᶻ, +∂ʸ)Ψ₂ 
    u₀(p, x, y, z) =
        Ψ₁(p, x, y, z) * (p.k * tan(p.k * y) + y / (p.l^2) + 1 / (10 * p.l))
    v₀(p, x, y, z) =
        Ψ₂(p, x, y, z) * (p.k * tan(p.k * z) + z / (p.l^2) + 1 / (10 * p.l)) -
        Ψ₁(p, x, y, z) * p.k * tan(p.k * x)
    w₀(p, x, y, z) = -Ψ₂(p, x, y, z) * p.k * tan(p.k * y)
    θ₀(p, x, y, z) = sin(p.k * y)

    ρ₀(p, x, y, z) = p.ρₒ
    ρu₀(p, x...) = ρ₀(p, x...) * (p.ϵ * u₀(p, x...) + U₀(p, x...))
    ρv₀(p, x...) = ρ₀(p, x...) * (p.ϵ * v₀(p, x...) + V₀(p, x...))
    ρw₀(p, x...) = ρ₀(p, x...) * (p.ϵ * w₀(p, x...) + W₀(p, x...))
    ρθ₀(p, x...) = ρ₀(p, x...) * θ₀(p, x...)

    ρu⃗₀(p, x...) = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀)

    for (key, grid) in grids
        @testset "$(key)" begin
            model = SpatialModel(
                balance_law = Fluid3D(),
                physics = physics,
                numerics = (flux = RoeNumericalFlux(),),
                grid = grid,
                boundary_conditions = NamedTuple(),
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
            evolve!(
                simulation,
                model;
                refDat = getproperty(refVals, Symbol(key)),
            )
        end
    end
end
