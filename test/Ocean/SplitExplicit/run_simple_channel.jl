#!/usr/bin/env julia --project
using Test

include("simple_channel_config.jl")
ClimateMachine.init()

const FT = Float64

#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/hydrostatic_spindown_refvals.jl")

    # simulation time
    timeend = FT(60 * 24 * 3600) # s
    tout = FT(24 * 3600) # s
    timespan = (tout, timeend)

    # DG polynomial order
    N = Int(4)

    # Domain resolution
    Nˣ = Int(4)
    Nʸ = Int(4)
    Nᶻ = Int(60)
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 1e6 / 12 # m
    Lʸ = 1e6 / 12 # m
    H = 3000  # m
    dimensions = (Lˣ, Lʸ, H)

    # model params
    cʰ = 1  # typical of ocean internal-wave speed 
    νʰ = 100.0
    νᶻ = 0.02
    κʰ = 100.0
    κᶻ = 0.02
    κᶜ = 0.1
    fₒ = -1e-4
    modelparams = (cʰ, νʰ, νᶻ, κʰ, κᶻ, κᶜ, fₒ)

    # problem params
    h = 1000 # m
    efl = 50e3 # e folding length 
    σʳ = 1 // (7 * 86400) # 1/s
    τₒ = 2e-1  # (Pa = N/m^2)
    λʳ = 10 // 86400 # m/s
    θᴱ = 10    # deg.C
    λᴰ = 1e-3   # drag coefficent, m/s
    problemparams = (h, efl, σʳ, τₒ, λʳ, θᴱ, λᴰ)

    config = SplitConfig(
        "channel_driver",
        resolution,
        dimensions,
        modelparams,
        problemparams,
        240,
    )

    run_split_explicit(
        config,
        timespan;
        dt_fast = 10, # seconds
        dt_slow = 240, # seconds
        # refDat = refVals.ninety_minutes,
        # analytic_solution = true,
    )
end
