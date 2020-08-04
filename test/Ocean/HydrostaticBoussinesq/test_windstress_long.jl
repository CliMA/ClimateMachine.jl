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
    timestart = FT(0)      # s
    timeend = FT(86400) # s
    timespan = (timestart, timeend)

    # DG polynomial order
    N = Int(4)

    # Domain resolution
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(50)
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 400   # m
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(NoSlip()), Insulating()),
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(KinematicStress()), Insulating()),
    )

    run_simple_box(
        "test_windstress_long_imex",
        resolution,
        dimensions,
        timespan,
        HomogeneousBox,
        imex = true,
        BC = BC,
        Δt = 55,
        refDat = refVals.long,
    )
end
