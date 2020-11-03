#!/usr/bin/env julia --project
using Test

include("hydrostatic_spindown.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/hydrostatic_spindown_refvals.jl")

    # simulation time
    timeend = FT(15 * 24 * 3600) # s
    tout = FT(24 * 3600) # s
    timespan = (tout, timeend)

    # DG polynomial order
    N = Int(4)

    # Domain resolution
    Nˣ = Int(5)
    Nʸ = Int(5)
    Nᶻ = Int(8)
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 1e6  # m
    Lʸ = 1e6  # m
    H = 400  # m
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    )
    config = SplitConfig(
        "rotating_bla",
        resolution,
        dimensions,
        Coupled(),
        Rotating();
        solver = SplitExplicitSolver,
        boundary_conditions = BC,
    )

    #=
    BC = (
        ClimateMachine.Ocean.SplitExplicit01.OceanFloorFreeSlip(),
        ClimateMachine.Ocean.SplitExplicit01.OceanSurfaceNoStressNoForcing(),
    )

    config = SplitConfig(
        "rotating_jmc",
        resolution,
        dimensions,
        Coupled(),
        Rotating();
        solver = SplitExplicitLSRK2nSolver,
        boundary_conditions = BC,
    )
    =#

    run_split_explicit(
        config,
        timespan,
        dt_fast = 300,
        dt_slow = 300, # 90 * 60,
        # refDat = refVals.ninety_minutes,
        analytic_solution = true,
    )
end
