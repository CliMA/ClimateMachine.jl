#!/usr/bin/env julia --project

ClimateMachine.init()

let
    # Domain size
    Ωˣ = Periodic(-2π, 2π)
    Ωʸ = Periodic(-2π, 2π)
    Ωᶻ = Periodic(-2π, 2π)
    Ω = Ωˣ × Ωʸ × Ωᶻ

    # Domain Resolution
    N = 1
    Nˣ = 8
    Nʸ = 8
    Nᶻ = 8
    resolution = (; N, Nˣ, Nʸ, Nᶻ)

    # model params
    cₛ = sqrt(10) # m/s
    ρₒ = 1 # kg/m³
    μ = 0 # 1e-6,   # m²/s
    ν = 1e-2   # m²/s
    κ = 1e-2   # m²/s
    α = 2e-4   # 1/K
    g = 10     # m/s²
    params = (; cₛ, ρₒ, μ, ν, κ, α, g)

    BC = (
        ClimateMachine.Ocean.OceanBC(Impenetrable(NoSlip()), Insulating()),
        ClimateMachine.Ocean.OceanBC(
            Impenetrable(KinematicStress(
                (state, aux, t) -> (@SVector [0.01 / state.ρ, -0, -0]),
            )),
            TemperatureFlux((state, aux, t) -> (0.1)),
        ),
    )

    config = Config(
        "heat_the_box",
        resolution,
        domain,
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 1,
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
        boundary_conditons = BC,
    )

    # simulation times
    timeend = FT(200) # s
    dt = FT(0.05) # s
    nout = Int(200)
    timespan = (; dt, nout, timeend)

    tic = Base.time()

    run_CNSE(config, resolution, timespan; TimeStepper = SSPRK22Heuns)

    toc = Base.time()
    time = toc - tic
    println(time)
end