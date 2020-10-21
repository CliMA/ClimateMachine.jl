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
    timeend = FT(24 * 3600) # s
    tout = FT(3 * 3600) # s
    dt_fast = 300 # seconds
    dt_slow = 90 * 60 # seconds

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

    solver = SplitExplicitSolverType{FT}(
        SplitExplicitSolver,
        dt_fast,
        dt_slow;
        add_fast_steps = 0,
        numImplSteps = 0,
    )

    config = SplitConfig("test_restart", resolution, dimensions, solver)

    midpoint = timeend / 2
    timespan = (tout, midpoint)

    run_split_explicit(config, midpoint, tout)

    run_split_explicit(
        config,
        midpoint,
        tout;
        refDat = refVals.ninety_minutes,
        analytic_solution = true,
        restart = Int(midpoint / tout),
    )
end
