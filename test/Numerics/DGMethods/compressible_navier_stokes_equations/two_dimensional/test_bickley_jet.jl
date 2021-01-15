#!/usr/bin/env julia --project

include("bickley_jet.jl")
ClimateMachine.init()

const FT = Float64
const vtkpath = nothing
#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("refvals_bickley_jet.jl")

    # simulation times
    timeend = FT(200) # s
    dt = FT(0.02) # s
    nout = Int(1000)

    # Domain Resolution
    N = 3
    Nˣ = 16
    Nʸ = 16

    # Domain size
    Lˣ = 4 * FT(π)  # m
    Lʸ = 4 * FT(π)  # m

    params = (; N, Nˣ, Nʸ, Lˣ, Lʸ, dt, nout, timeend)

    setups = [
        (;
            name = "rusanov_periodic",
            flux = RusanovNumericalFlux(),
            periodicity = (true, true),
            boundary = ((0, 0), (0, 0)),
            Nover = 0,
        ),
        (;
            name = "roeflux_periodic",
            flux = RoeNumericalFlux(),
            periodicity = (true, true),
            boundary = ((0, 0), (0, 0)),
            Nover = 0,
        ),
        (;
            name = "rusanov",
            flux = RusanovNumericalFlux(),
            periodicity = (true, false),
            boundary = ((0, 0), (1, 1)),
            Nover = 0,
        ),
        (;
            name = "roeflux",
            flux = RoeNumericalFlux(),
            periodicity = (true, false),
            boundary = ((0, 0), (1, 1)),
            Nover = 0,
        ),
        (;
            name = "rusanov_overintegration",
            flux = RusanovNumericalFlux(),
            periodicity = (true, false),
            boundary = ((0, 0), (1, 1)),
            Nover = 1,
        ),
        # rusanov and overintegration seems to be non-deterministic
        (;
            name = "roeflux_overintegration",
            flux = RoeNumericalFlux(),
            periodicity = (true, false),
            boundary = ((0, 0), (1, 1)),
            Nover = 1,
        ),
    ]

    for setup in setups
        @testset "$(setup.name)" begin
            config = Config(
                setup.name,
                params;
                numerical_flux_first_order = setup.flux,
                Nover = setup.Nover,
                periodicity = setup.periodicity,
                boundary = setup.boundary,
                boundary_conditons = (ClimateMachine.Ocean.OceanBC(
                    Impenetrable(FreeSlip()),
                    Insulating(),
                ),),
            )

            run_bickley_jet(
                config,
                params;
                refDat = getproperty(refVals, Symbol(setup.name)),
            )
        end
    end

end
