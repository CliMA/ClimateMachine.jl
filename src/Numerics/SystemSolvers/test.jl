using Test
using ClimateMachine.SystemSolvers

using StaticArrays, LinearAlgebra, Random

struct TempStruct
    f!
end
(ts::TempStruct)(args...) = ts.f!(args...)

@testset "Newton-GMRES" begin
    Random.seed!(1)

    for T in [Float32, Float64]
        A = rand(T, 200, 1000)
        A = 1e-2 * A' * A + I
        b = rand(T, 1000)

        mulbyA!(y, x) = (y .= A * x)

        x0 = rand(T, 1000)
        x1 = copy(x0)
        x2 = copy(x0)
#=
        originalsolver = GeneralizedMinimalResidual(b, M = 20)
        newsolver = IterativeSolver(GeneralizedMinimalResidualAlgorithm(M = 20), StandardProblem(mulbyA!, x2, b))

        linearsolve!(mulbyA!, nothing, originalsolver, x1, b)
        solve!(newsolver, StandardProblem(mulbyA!, x2, b))

        @test x1 == x2
=#
    end

    for T in [Float32, Float64]
        f!(fx, x) = (fx[1] = x[1]^2 + T(5) / x[2]^4 + T(2) * x[3]; fx[2] = (x[1] - T(1))^3 + T(3) * x[2]; fx[3] = T(4) * sin(x[1] * x[2] * x[3]))
        tsf! = TempStruct(f!)
        x0 = rand(T, 3)
        x1 = copy(x0)
        x2 = copy(x0)
        x3 = copy(x0)
        b = T[1, 2, 3]

        originalsolver = JacobianFreeNewtonKrylovSolver(x1, GeneralizedMinimalResidual(x1, M = 3))
        originaljvp = JacobianAction(tsf!, x1, originalsolver.ϵ)
        newsolver = IterativeSolver(JacobianFreeNewtonKrylovAlgorithm(GeneralizedMinimalResidualAlgorithm(M = 3); maxiters = 10000), StandardProblem(f!, x2, b))
        newsolver2 = IterativeSolver(JacobianFreeNewtonKrylovAlgorithm(GeneralizedMinimalResidualAlgorithm(M = 3); maxiters = 10000, autodiff = true), StandardProblem(f!, x3, b))

        println("Printing stuff for original Newton----------------------------------")
        println("iters = ", nonlinearsolve!(tsf!, originaljvp, nothing, originalsolver, x1, b, max_newton_iters = 10000))
        println("Printing stuff for new Newton FD------------------------------------")
        println("iters = ", solve!(newsolver, StandardProblem(f!, x2, b)))
        println("Printing stuff for new Newton AD------------------------------------")
        println("iters = ", solve!(newsolver2, StandardProblem(f!, x3, b)))

        println(x1)
        println(x2)
        println(x3)
        @test x1 == x2
    end
end