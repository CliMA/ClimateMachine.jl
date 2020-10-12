#!/usr/bin/env julia --project
using Test

include("ocean_gyre.jl")
ClimateMachine.init(disable_gpu = true)

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    # include("../refvals/hydrostatic_spindown_refvals.jl")

    # simulation time
    timeend = FT(15 * 24 * 3600) # s
    tout = FT(16 * 90 * 60) # s
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

    problem = OceanGyre{FT}(Lˣ, Lʸ, H)

    model_3D = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem;
        coupling = Coupled(),
        tracer_advection = nothing,
        cʰ = FT(1),
        κʰ = FT(0),
        κᶻ = FT(0),
        κᶜ = FT(0),
    )

    config = SplitConfig("test_SST", resolution, dimensions, model_3D)

    run_split_explicit(
        config,
        timespan;
        dt_fast = 300,
        dt_slow = 90 * 60,
        additional_percent = 0.5,
        # refDat = refVals.ninety_minutes,
    )
end
