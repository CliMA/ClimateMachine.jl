using Test
using ClimateMachine
using ClimateMachine.SystemSolvers

using StaticArrays, SparseArrays, LinearAlgebra, Random

@testset "Iterative Solver Parameters" begin
    Random.seed!(42)
    algs = (
        (atol, rtol, maxiters) -> GeneralizedMinimalResidualAlgorithm(;
            atol = atol, rtol = rtol, maxrestarts = maxiters - 1,
        ),
        (atol, rtol, maxiters) -> BatchedGeneralizedMinimalResidualAlgorithm(;
            atol = atol, rtol = rtol, maxrestarts = maxiters - 1,
        ),
        (atol, rtol, maxiters) -> GeneralizedConjugateResidualAlgorithm(;
            atol = atol, rtol = rtol, maxrestarts = maxiters - 1,
        ),
        (atol, rtol, maxiters) -> ConjugateGradientAlgorithm(;
            atol = atol, rtol = rtol, maxiters = maxiters,
        ),
        (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(GeneralizedMinimalResidualAlgorithm();
            atol = atol, rtol = rtol, maxiters = maxiters,
        ),
        (atol, rtol, maxiters) -> StandardPicardAlgorithm(
            atol = atol, rtol = rtol, maxiters = maxiters,
        ),
        (atol, rtol, maxiters) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(atol = atol, rtol = rtol, maxiters = maxiters,),
        ),
    )
    @testset "atol, rtol, maxiters functions" begin
        # test rtol, atol, maxiters functions return the correct type and value

        f(args...) = nothing

        params = (
            (atol = 1.e-4, rtol = 1.e-3, maxiters = 14),
            (atol = 1.f-4, rtol = 1.f-3, maxiters = 12),
            (atol = 1.e-4, rtol = 1.f-3, maxiters = 14),
            (atol = 1.f-4, rtol = 1.e-3, maxiters = 12),
        )
        
        for alg in algs
            for FT in [Float32, Float64]
                for p in params
                    Q = zeros(FT, 10)
                    rhs = zeros(FT, 10)
                    solver = IterativeSolver(alg(p.atol, p.rtol, p.maxiters), Q, f, rhs)
                    @test FT(p.atol) == ClimateMachine.SystemSolvers.atol(solver)
                    @test FT(p.rtol) == ClimateMachine.SystemSolvers.rtol(solver)
                    @test p.maxiters == ClimateMachine.SystemSolvers.maxiters(solver)
                end
            end
        end
    end

    @testset "atol, rtol, maxiters Parameter Domains" begin
        params = (
            (atol = -1.e-4, rtol = 1.e-5, maxiters = 14),
            (atol = 1.e-4, rtol = -1.e-5, maxiters = 14),
            (atol = 1.e-4, rtol = 1.e-5, maxiters = -14),
            (atol = 0.0, rtol = 1.e-5, maxiters = 14),
            (atol = 1.e-4, rtol = 0.0, maxiters = 14),
            (atol = 1.e-4, rtol = 1.e-5, maxiters = 0),
        )

        for alg in algs
            for p in params
                @test_throws DomainError alg(p.atol, p.rtol, p.maxiters)
            end
        end
    end

    @testset "Restarted Method Parameter Domains" begin
        algs = (
            (maxrestarts, M, groupsize) -> GeneralizedMinimalResidualAlgorithm(
                maxrestarts = maxrestarts, M = M, groupsize = groupsize,
            ),
            (maxrestarts, M, groupsize) -> GeneralizedConjugateResidualAlgorithm(
                maxrestarts = maxrestarts, M = M, groupsize = groupsize,
            ),
            (maxrestarts, M, groupsize) -> BatchedGeneralizedMinimalResidualAlgorithm(
                maxrestarts = maxrestarts, M = M, groupsize = groupsize,
            ),
        )
        params = (
            (maxrestarts = -1, M = 10, groupsize = 128),
            (maxrestarts = 1, M = -10, groupsize = 128),
            (maxrestarts = 1, M = 10, groupsize = -128),
            (maxrestarts = 1, M = 0, groupsize = 128),
            (maxrestarts = 1, M = 10, groupsize = 0),
        )

        # @test_throws DomainError


        # @test maxrestarts * M == maxiters(solver)
    end

    @testset "Batched GMRES Parameter Domains" begin

    end

    @testset "Newton β Parameter Domain" begin
        algs = (
            (β) -> JacobianFreeNewtonKrylovAlgorithm(
                    GeneralizedMinimalResidualAlgorithm();
                    autodiff = true, β = β,
                ),
            (β) -> JacobianFreeNewtonKrylovAlgorithm(
                GeneralizedMinimalResidualAlgorithm();
                autodiff = false, β = β,
                ),
        )

        βs = (-1.0, 0.0, Inf)

        for alg in algs
            for β in βs
                @test_throws DomainError alg(β)
            end
        end
    end

    @testset "Acceleration Algorithm Parameter Domains" begin
        algs = (
            (depth, ω) -> AndersonAccelerationAlgorithm(StandardPicardAlgorithm();
                depth = depth, ω = ω,
            ),
        )

        params = (
            (depth = 0, ω = 0.5),
            (depth = -1, ω = 0.5),
            (depth = 1, ω = 0.0),
            (depth = 1, ω = -0.5),
            (depth = 1, ω = 1.1),
        )

        for alg in algs
            for p in params
                @test_throws DomainError alg(p.depth, p.ω)
            end
        end
    end

    @testset "Square system checks" begin

    end
end
