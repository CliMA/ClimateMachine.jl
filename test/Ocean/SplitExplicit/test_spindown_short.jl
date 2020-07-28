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

    @testset "Multi-rate" begin
        @testset "Δt = 30 mins" begin
            run_hydrostatic_spindown(
                "vtk_split",
                resolution,
                dimensions,
                timespan,
                coupling = Coupled(),
                dt_slow = 30 * 60,
                refDat = refVals.thirty_minutes,
            )
        end

        @testset "Δt = 60 mins" begin
            run_hydrostatic_spindown(
                "vtk_split",
                resolution,
                dimensions,
                timespan,
                coupling = Coupled(),
                dt_slow = 60 * 60,
                refDat = refVals.sixty_minutes,
            )
        end

        @testset "Δt = 90 mins" begin
            run_hydrostatic_spindown(
                "vtk_split",
                resolution,
                dimensions,
                timespan,
                coupling = Coupled(),
                dt_slow = 90 * 60,
                refDat = refVals.ninety_minutes,
            )
        end
    end
end
