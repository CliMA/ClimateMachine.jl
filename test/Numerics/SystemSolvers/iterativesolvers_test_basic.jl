using Test
using ClimateMachine
using ClimateMachine.SystemSolvers

using StaticArrays, LinearAlgebra, Random

Random.seed!(1)

# @testset "Solver Params" begin
#     # test rtol, atol, maxiter functions return the correct type and value
#     # test errors throw for out of bounds values

#     FT = Float32
#     f(args...) = nothing
#     Q = [FT(1)]
#     rhs = [FT(1)]

#     algs = (
#         (atol, rtol, maxiters) -> GeneralizedMinimalResidualAlgorithm(;
#             atol = atol,
#             rtol = rtol,
#             maxrestarts = maxiters,
#         ),
#         (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
#             GeneralizedMinimalResidualAlgorithm();
#             atol = atol,
#             rtol = rtol,
#             maxiters = maxiters,
#         ),
#         (atol, rtol, maxiters) -> StandardPicardAlgorithm(;
#             atol = atol,
#             rtol = rtol,
#             maxiters = maxiters,
#         ),
#     )

#     atol, rtol, maxiters = 1.e-4, 1.e-5, 14

#     for alg in algs
#         solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f, rhs)
#         a = ClimateMachine.SystemSolvers.atol(solver)
#         r = ClimateMachine.SystemSolvers.rtol(solver)
#         m = ClimateMachine.SystemSolvers.maxiters(solver)
#         @test a == FT(atol)
#         @test r == FT(rtol)
#         @test m == maxiters
#     end

#     params = (
#         (-1.e-4, 1.e-5, 14),
#         (1.e-4, -1.e-5, 14),
#         (1.e-4, 1.e-5, -14),
#     )
#     for alg in algs
#         for (atol, rtol, maxiters) in params
#             @test_throws DomainError alg(atol, rtol, maxiters)
#         end
#     end
# end

# @testset "Standard Problems" begin
#     algs = (
#             (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
#                 GeneralizedMinimalResidualAlgorithm()),
#             (atol, rtol, maxiters) -> JacobianFreeNewtonKrylovAlgorithm(
#                 GeneralizedMinimalResidualAlgorithm(); autodiff=true),
#         )

#     atol, rtol, maxiters = 1.e-6, 1.e-6, 10

#     for T in [Float32, Float64]
#         f!(fx, x) = (fx[1] = x[1]^2 + T(2); fx[2] = (x[2] - T(1))^3 + T(3))
#         x0 = rand(T, 2)
#         b = T[11, 11]
#         x_true = T[3, 3]

#         for alg in algs
#             Q = copy(x0)
#             solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f!, b)
#             solve!(solver, Q, f!, b)
#             @test norm(Q - x_true) < 1.e-6
#         end
#     end
# end

@testset "Fixed Point Problems" begin
    algs = (
        (atol, rtol, maxiters) -> StandardPicardAlgorithm(;
            atol = atol,
            rtol = rtol,
            maxiters = maxiters,
        ),
        (atol, rtol, maxiters) -> AndersonAccelerationAlgorithm(
            StandardPicardAlgorithm(; atol = atol, rtol = rtol, maxiters = maxiters);
            depth = 4,
        )
    )    

    atol, rtol, maxiters = 1.e-6, 1.e-6, 40

    for T in [Float32, Float64]
        function f!(y,x)
            y[1] = .5(exp(-x[1]) + x[2])
            y[2] = (exp(-x[2]) + x[1])/3
        end
        x0 = rand(T, 2)
        x_true = T[0.499336, 0.391739]
        # f!(y,x) = (y .= x./T(10))
        # x0 = T[.5]
        # x_true = T[0.0]
        
        for alg in algs 
            Q = copy(x0)
            solver = IterativeSolver(alg(atol, rtol, maxiters), Q, f!)
            iters, fcalls = solve!(solver, Q, f!)
            @info iters, fcalls
            @info Q
            @test norm(Q - x_true) < atol
        end
    end
end