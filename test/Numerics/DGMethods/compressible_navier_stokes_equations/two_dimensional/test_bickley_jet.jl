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
    nout = Int(timeend / dt / 10)

    # Domain Resolution
    N = 3
    Nˣ = 16
    Nʸ = 16

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
                resolution,
                domain,
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

            run_CNSE(
                config,
                resolution,
                timespan;
                refDat = getproperty(refVals, Symbol(setup.name)),
            )
        end
    end

end
