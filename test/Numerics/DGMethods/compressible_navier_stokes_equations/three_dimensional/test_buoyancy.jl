#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("refvals_buoyancy.jl")

    ########
    # Setup physical and numerical domains
    ########
    Ωˣ = IntervalDomain(-2π, 2π, periodic = true)
    Ωʸ = IntervalDomain(-2π, 2π, periodic = true)
    Ωᶻ = IntervalDomain(0, 4π, periodic = false)

    second_order = DiscretizedDomain(
        Ωˣ × Ωʸ × Ωᶻ;
        elements = 5,
        polynomial_order = 2,
        overintegration_order = 1,
    )

    fourth_order = DiscretizedDomain(
        Ωˣ × Ωʸ × Ωᶻ;
        elements = 3,
        polynomial_order = 4,
        overintegration_order = 1,
    )

    grids = Dict("second_order" => second_order, "fourth_order" => fourth_order)

    ########
    # Define timestepping parameters
    ########
    start_time = 0
    end_time = 0.1
    Δt = 0.001
    method = SSPRK22Heuns

    timestepper = TimeStepper(method = method, timestep = Δt)

    callbacks = (
        Info(),
        # VTKState(; iteration = 1, filepath = "output/buoyancy"),
        StateCheck(10),
    )

    ########
    # Define physical parameters and parameterizations
    ########
    Lˣ, Lʸ, Lᶻ = length(Ωˣ × Ωʸ × Ωᶻ)
    parameters = (
        ρₒ = 1, # reference density
        cₛ = sqrt(10), # sound speed
        α = 1e-4, # thermal expansion coefficient
        g = 10, # gravity
        θₒ = 10, # initial temperature value
        Lˣ = Lˣ,
        Lʸ = Lʸ,
        H = Lᶻ,
    )

    physics = FluidPhysics(;
        advection = NonLinearAdvectionTerm(),
        dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0, κ = 0),
        coriolis = nothing,
        buoyancy = Buoyancy{Float64}(α = parameters.α, g = parameters.g),
    )

    ########
    # Define initial conditions
    ########

    u₀(p, x, y, z) = -0
    v₀(p, x, y, z) = -0
    w₀(p, x, y, z) = -0
    θ₀(p, x, y, z) = -p.θₒ * (1 - z / 4π)

    ρ₀(p, x, y, z) =
        p.ρₒ * (1 - (p.α * p.g / p.cₛ^2) / 2 * (-p.θₒ * (1 - z / 4π))^2)
    ρu₀(p, x...) = ρ₀(p, x...) * u₀(p, x...)
    ρv₀(p, x...) = ρ₀(p, x...) * v₀(p, x...)
    ρw₀(p, x...) = ρ₀(p, x...) * w₀(p, x...)
    ρθ₀(p, x...) = ρ₀(p, x...) * θ₀(p, x...)

    ρu⃗₀(p, x...) = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀)

    orientations = Dict(
        "" => ClimateMachine.Orientations.NoOrientation(),
        "_flat" => ClimateMachine.Orientations.FlatOrientation(),
    )

    for (key1, grid) in grids
        for (key2, orientation) in orientations
            key = key1 * key2

            println("running ", key)

            local_physics = FluidPhysics(;
                orientation = orientation,
                advection = physics.advection,
                dissipation = physics.dissipation,
                coriolis = physics.coriolis,
                buoyancy = physics.buoyancy,
            )

            @testset "$(key)" begin
                model = SpatialModel(
                    balance_law = Fluid3D(),
                    physics = local_physics,
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
                    model,
                    refDat = getproperty(refVals, Symbol(key)),
                )
            end
        end
    end
end
