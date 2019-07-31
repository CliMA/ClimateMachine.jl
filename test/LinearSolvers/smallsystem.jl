using Test
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver
using CLIMA.GeneralizedMinimalResidualSolver

using StaticArrays, LinearAlgebra, Random

# this test setup is partly based on IterativeSolvers.jl, see e.g
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/test/cg.jl
@testset "LinearSolvers small full system" begin
  n = 10

  methods = ((b, tol) -> GeneralizedConjugateResidual(2, b, tol),
             (b, tol) -> GeneralizedMinimalResidual(3, b, tol),
             (b, tol) -> GeneralizedMinimalResidual(n, b, tol)
            )

  expected_iters = (Dict(Float32 => 7, Float64 => 11),
                    Dict(Float32 => 9, Float64 => 21),
                    Dict(Float32 => 6, Float64 => 10)
                   )

  for (m, method) in enumerate(methods), T in [Float32, Float64]
    Random.seed!(44)

    A = @MMatrix rand(T, n, n)
    A = A' * A + I
    b = @MVector rand(T, n)

    mulbyA!(y, x) = (y .= A * x)

    tol = sqrt(eps(T))
    linearsolver = method(b, tol)
    
    x = @MVector rand(T, n)
    iters = linearsolve!(mulbyA!, x, b, linearsolver)

    @test iters == expected_iters[m][T]
    @test norm(A * x - b) / norm(b) <= tol
   
    newtol = 1000tol
    settolerance!(linearsolver, newtol)
    
    x = @MVector rand(T, n)
    linearsolve!(mulbyA!, x, b, linearsolver)

    @test norm(A * x - b) / norm(b) <= newtol
    @test norm(A * x - b) / norm(b) >= tol

  end
end
