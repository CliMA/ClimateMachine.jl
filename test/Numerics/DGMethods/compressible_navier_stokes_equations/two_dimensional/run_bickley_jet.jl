#!/usr/bin/env julia --project

include("bickley_jet.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################

vtkpath = abspath(joinpath(
    ClimateMachine.Settings.output_dir,
    "vtk_roe_flux_analytic_fixed",
))

let
    # simulation times
    timeend = FT(200) # s
    dt = FT(0.02) # s
    nout = Int(100)

    # Domain Resolution
    N = 3
    Nˣ = 16
    Nʸ = 16

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m

    params = (; N, Nˣ, Nʸ, Lˣ, Lʸ, dt, nout, timeend)

    config = Config(
        "bickley_jet",
        params;
        numerical_flux_first_order = RoeNumericalFlux(),
        Nover = 0,
        periodicity = (true, false),
        boundary = ((0, 0), (1, 1)),
        boundary_conditons = (ClimateMachine.Ocean.OceanBC(
            Impenetrable(FreeSlip()),
            Insulating(),
        ),),
    )

    tic = Base.time()

    run_bickley_jet(config, params; TimeStepper = LSRK54CarpenterKennedy)

    toc = Base.time()
    time = toc - tic
    println(time)
end
