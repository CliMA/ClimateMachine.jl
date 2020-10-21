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

    @testset "Single-Rate" begin
        solver = SplitExplicitSolverType{FT}(
            SplitExplicitSolver,
            300,
            300;
            add_fast_steps = 0,
            numImplSteps = 0,
        )

        @testset "Not Coupled" begin
            config = SplitConfig(
                "uncoupled",
                resolution,
                dimensions,
                solver;
                coupling = Uncoupled(),
            )

            run_split_explicit(
                config,
                timeend,
                tout;
                refDat = refVals.uncoupled,
                analytic_solution = true,
            )
        end

        @testset "Fully Coupled" begin
            config = SplitConfig("coupled", resolution, dimensions, solver)

            run_split_explicit(
                config,
                timeend,
                tout;
                refDat = refVals.coupled,
                analytic_solution = true,
            )
        end
    end

    @testset "Multi-rate" begin
        @testset "Δt = 30 mins" begin
            solver = SplitExplicitSolverType{FT}(
                SplitExplicitSolver,
                300,
                30 * 60;
                add_fast_steps = 0,
                numImplSteps = 0,
            )

            config =
                SplitConfig("thirty_minutes", resolution, dimensions, solver)

            run_split_explicit(
                config,
                timeend,
                tout;
                refDat = refVals.thirty_minutes,
                analytic_solution = true,
            )
        end

        @testset "Δt = 60 mins" begin
            solver = SplitExplicitSolverType{FT}(
                SplitExplicitSolver,
                300,
                60 * 60;
                add_fast_steps = 0,
                numImplSteps = 0,
            )

            config =
                SplitConfig("sixty_minutes", resolution, dimensions, solver)

            run_split_explicit(
                config,
                timeend,
                tout;
                refDat = refVals.sixty_minutes,
                analytic_solution = true,
            )
        end

        @testset "Δt = 90 mins" begin
            solver = SplitExplicitSolverType{FT}(
                SplitExplicitSolver,
                300,
                90 * 60;
                add_fast_steps = 0,
                numImplSteps = 0,
            )

            config =
                SplitConfig("ninety_minutes", resolution, dimensions, solver)

            run_split_explicit(
                config,
                timeend,
                tout;
                refDat = refVals.ninety_minutes,
                analytic_solution = true,
            )
        end
    end
end
