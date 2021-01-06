using Test
using ClimateMachine.SystemSolvers

using StaticArrays, LinearAlgebra, Random

struct TempStruct
    f!
end
(ts::TempStruct)(args...) = ts.f!(args...)

@testset "Original vs. New" begin
    Random.seed!(1)

    for T in [Float32, Float64], _ in 1:10
        A = rand(T, 200, 250)
        A = T(10) * rand(T) * A' * A + I
        f!(y, x) = (
            mul!(view(y, 1:250), A, view(x, 1:250));
            mul!(view(y, 251:500), A, view(x, 251:500));
            mul!(view(y, 501:750), A, view(x, 501:750));
            mul!(view(y, 751:1000), A, view(x, 751:1000));
            y[1:250] .*= A[1]; y[251:500] .*= A[2];
            y[501:750] .*= A[3]; y[751:1000] .*= A[4];
        )
        x0 = rand(T, 1000)
        x1 = copy(x0)
        x2 = copy(x0)
        x3 = copy(x0)
        x4 = copy(x0)
        x5 = copy(x0)
        x6 = copy(x0)
        b = rand(T, 1000)

        originalsolver1 = GeneralizedMinimalResidual(x0, M = 20)
        originalsolver2 = BatchedGeneralizedMinimalResidual(x0, 500, 2, forward_reshape = (500, 2), forward_permute = (1, 2))
        originalsolver3 = BatchedGeneralizedMinimalResidual(x0, 250, 4, forward_reshape = (250, 4), forward_permute = (1, 2))
        newsolver1 = IterativeSolver(GeneralizedMinimalResidualAlgorithm(), x0, f!, b)
        newsolver2 = IterativeSolver(BatchedGeneralizedMinimalResidualAlgorithm(dims = (500, 2), batchdimindices = (1,)), x0, f!, b)
        newsolver3 = IterativeSolver(BatchedGeneralizedMinimalResidualAlgorithm(dims = (250, 4), batchdimindices = (1,)), x0, f!, b)

        iters1 = linearsolve!(f!, nothing, originalsolver1, x1, b)
        iters2 = linearsolve!(f!, nothing, originalsolver2, x2, b)
        iters3 = linearsolve!(f!, nothing, originalsolver3, x3, b)
        iters4 = solve!(newsolver1, x4, f!, b)
        iters5 = solve!(newsolver2, x5, f!, b)
        iters6 = solve!(newsolver3, x6, f!, b)

        println("Linear ($T): $iters1, $iters2, $iters3, $iters4, $iters5, $iters6")
        @test x1 == x4
        @test x2 == x5
        @test x3 == x6
        @test iters4[2] == iters1 + iters4[1]
        @test iters5[2] == iters2 + iters5[1]
        @test iters6[2] == iters3 + iters6[1]
    end

    for T in [Float32, Float64], _ in 1:10
        f!(fx, x) = (fx[1] = x[1]^2 - T(0.2) * x[3]; fx[2] = x[1]^3 - T(3) * x[2]; fx[3] = x[3]^2)
        tsf! = TempStruct(f!)
        x0 = rand(T, 3)
        x1 = copy(x0)
        x2 = copy(x0)
        x3 = copy(x0)
        b = rand(T, 3)

        originalsolver = JacobianFreeNewtonKrylovSolver(x1, GeneralizedMinimalResidual(x0, M = 20))
        originaljvp = JacobianAction(tsf!, x0, originalsolver.ϵ)
        newsolver1 = IterativeSolver(JacobianFreeNewtonKrylovAlgorithm(GeneralizedMinimalResidualAlgorithm()), x0, f!, b)
        newsolver2 = IterativeSolver(JacobianFreeNewtonKrylovAlgorithm(GeneralizedMinimalResidualAlgorithm(), autodiff = true), x0, f!, b)

        iters1 = nonlinearsolve!(tsf!, originaljvp, nothing, originalsolver, x1, b, max_newton_iters = 10000)
        iters2 = solve!(newsolver1, x2, f!, b)
        iters3 = solve!(newsolver2, x3, f!, b)

        println("Nonlinear ($T): $iters1, $iters2, $iters3")
        @test x1 == x2
        @test iters1 == iters2[1]
        @test norm(x2 .- x3) < newsolver1.atol
    end
end