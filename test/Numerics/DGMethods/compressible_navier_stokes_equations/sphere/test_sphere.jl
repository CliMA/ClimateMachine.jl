#!/usr/bin/env julia --project

include("../shared_source/boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")

ClimateMachine.init()

#! format: off
refVals = (
[
 [ "state",     "ρ",   9.99999999999999000799e-01,  1.00000000000000066613e+00,  1.00000000000000000000e+00,  3.49433608871729305050e-16 ],
 [ "state", "ρu[1]",  -1.03159890504820785885e-17,  9.46342010159792552240e-18, -5.38891312633679647218e-19,  2.05248736591058928424e-18 ],
 [ "state", "ρu[2]",  -1.12556410356665774824e-17,  1.11125313793689735132e-17, -2.17938113394074827612e-22,  2.14911076688864467562e-18 ],
 [ "state", "ρu[3]",  -1.12757405914899880325e-17,  9.66517285069783167720e-18, -2.95076506235937354400e-19,  2.21386287201671452859e-18 ],
 [ "state",    "ρθ",   9.99999999999999000799e-01,  1.00000000000000066613e+00,  1.00000000000000000000e+00,  3.49433608871729305050e-16 ],
],
[
 [ "state",     "ρ",    12,    12,    12,     0 ],
 [ "state", "ρu[1]",     0,     0,     0,     0 ],
 [ "state", "ρu[2]",     0,     0,     0,     0 ],
 [ "state", "ρu[3]",     0,     0,     0,     0 ],
 [ "state",    "ρθ",    12,    12,    12,     0 ],
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
    Ω = AtmosDomain(radius = 1, height = 0.02)
    grid = DiscretizedDomain(
        Ω;
        elements = (vertical = 1, horizontal = 4),
        polynomial_order = (vertical = 1, horizontal = 4),
        overintegration_order = 1,
    )

    ########
    # Define timestepping parameters
    ########
    start_time = 0
    end_time = 2.0
    Δt = 0.05
    method = SSPRK22Heuns
    timestepper = TimeStepper(method = method, timestep = Δt)
    callbacks = (Info(), StateCheck(10))

    ########
    # Define physical parameters and parameterizations
    ########
    parameters = (
        ρₒ = 1, # reference density
        cₛ = 1e-2, # sound speed
    )

    physics = FluidPhysics(;
        advection = NonLinearAdvectionTerm(),
        dissipation = ConstantViscosity{Float64}(μ = 0, ν = 0.0, κ = 0.0),
        coriolis = nothing,
        buoyancy = Buoyancy{Float64}(α = 2e-4, g = 0),
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
    ρ₀(p, x, y, z) = p.ρₒ
    ρu₀(p, x...) = ρ₀(p, x...) * -0
    ρv₀(p, x...) = ρ₀(p, x...) * -0
    ρw₀(p, x...) = ρ₀(p, x...) * -0
    ρθ₀(p, x...) = ρ₀(p, x...) * 1.0

    ρu⃗₀(p, x...) = @SVector [ρu₀(p, x...), ρv₀(p, x...), ρw₀(p, x...)]
    initial_conditions = (ρ = ρ₀, ρu = ρu⃗₀, ρθ = ρθ₀)

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
    evolve!(simulation, model; refDat = refVals)
    toc = Base.time()
    time = toc - tic
    println(time)

end
