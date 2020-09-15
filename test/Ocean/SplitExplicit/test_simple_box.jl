
include("../../../experiments/OceanSplitExplicit/simple_box.jl")
ClimateMachine.init()

# Float type
const FT = Float64


#################
# RUN THE TESTS #
#################
@testset "$(@__FILE__)" begin

    include("../refvals/simple_box_ivd_refvals.jl")
    refDat = (refVals[1], refPrecs[1])

    # simulation time
    timestart = FT(0)      # s
    timeend = FT(5 * 86400) # s
    timespan = (timestart, timeend)

    # DG polynomial order
    N = Int(4)

    # Domain resolution
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(20)
    resolution = (N, Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    config = config_simple_box(
        "test_simple_box",
        resolution,
        dimensions;
        dt_slow = FT(90 * 60),
        dt_fast = FT(240),
    )

    run_simple_box(config, timespan; refDat = refDat)

end
