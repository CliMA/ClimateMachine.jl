#!/usr/bin/env julia --project

include("../../../experiments/OceanBoxGCM/simple_box.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/test_windstress_refvals.jl")

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
    H = 400   # m
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(KinematicStress()), Insulating()),
    )

    run_simple_box(
        "test_windstress_short_imex",
        resolution,
        dimensions,
        timespan,
        HomogeneousBox,
        imex = true,
        BC = BC,
        Δt = 60,
        refDat = refVals.imex,
    )

    run_simple_box(
        "test_windstress_short_explicit",
        resolution,
        dimensions,
        timespan,
        HomogeneousBox,
        imex = false,
        BC = BC,
        Δt = 180,
        refDat = refVals.explicit,
    )
end
