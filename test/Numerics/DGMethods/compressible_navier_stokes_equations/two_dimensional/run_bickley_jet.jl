#!/usr/bin/env julia --project

include("bickley_jet.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################

vtkpath =
    abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk_bickley_2D"))

let
    # simulation times
    timeend = FT(200) # s
    dt = FT(0.02) # s
    nout = Int(100)

    # Domain Resolution
    N = 3
    Nˣ = 8
    Nʸ = 8

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m

    # model params
    c = 2 # m/s
    g = 10 # m/s²
    ν = 0 # 1e-6,   # m²/s
    κ = 0 # 1e-6,   # m²/s

    resolution = (; N, Nˣ, Nʸ)
    domain = (; Lˣ, Lʸ)
    timespan = (; dt, nout, timeend)
    params = (; c, g, ν, κ)

    config = Config(
        "bickley_jet",
        resolution,
        domain,
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 1,
        periodicity = (true, false),
        boundary = ((0, 0), (1, 1)),
        boundary_conditons = (ClimateMachine.Ocean.OceanBC(
            Impenetrable(FreeSlip()),
            Insulating(),
        ),),
    )

    tic = Base.time()

    run_CNSE(config, resolution, timespan; TimeStepper = SSPRK22Heuns)

    toc = Base.time()
    time = toc - tic
    println(time)
end
