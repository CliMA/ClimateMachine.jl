#!/usr/bin/env julia --project

include("../../../experiments/OceanBoxGCM/simple_box.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/test_ocean_gyre_refvals.jl")

    # simulation time
    timestart = FT(0)    # s
    timeend = FT(3600)  # s
    timespan = (timestart, timeend)

    # DG polynomial order
    N = Int(4)

    # Domain resolution
    Nˣ = Int(5)
    Nʸ = Int(5)
    Nᶻ = Int(5)
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 1e6    # m
    Lʸ = 1e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
    )

    run_simple_box(
        "ocean_gyre_short",
        resolution,
        dimensions,
        timespan,
        OceanGyre,
        imex = false,
        BC = BC,
        Δt = 120,
        refDat = refVals.short,
    )
end
