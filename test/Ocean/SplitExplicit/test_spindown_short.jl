#!/usr/bin/env julia --project

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
    tout = FT(1.5 * 3600) # s
    timespan = (tout, timeend)

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

    BC = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()),
        OceanBC(Penetrable(FreeSlip()), Insulating()),
    )

    solver = SplitExplicitSolverType{FT}(
        SplitExplicitSolver,
        dt_fast,
        dt_slow;
        add_fast_steps = 0,
        numImplSteps = 0,
    )

    config = SplitConfig(
        "spindown_bla",
        resolution,
        dimensions,
        Coupled();
        solver = solver,
        boundary_conditions = BC,
    )

    run_split_explicit(
        config,
        timeend,
        tout;
        # refDat = refVals.ninety_minutes,
        analytic_solution = true,
    )

    #=
    BC = (
        ClimateMachine.Ocean.SplitExplicit01.OceanFloorFreeSlip(),
        ClimateMachine.Ocean.SplitExplicit01.OceanSurfaceNoStressNoForcing(),
    )

    solver = SplitExplicitSolverType{FT}(
        SplitExplicitLSRK2nSolver,
        dt_fast,
        dt_slow;
        add_fast_steps = 2,
        numImplSteps = 5,
    )

    config = SplitConfig(
        "spindown_jmc",
        resolution,
        dimensions,
        Coupled();
        solver = solver,
        boundary_conditions = BC,
    )

    run_split_explicit(
        config,
        timeend,
        tout;
        # refDat = refVals.ninety_minutes,
        analytic_solution = false,
    )
    =#
end
