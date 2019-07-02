using Test
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver

using LinearAlgebra, Random


# this test setup is partly based on IterativeSolvers.jl, see e.g
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/test/cg.jl
@testset "LinearSolvers small full system" begin
  n = 10

  expected_iters = Dict(Float32 => 4, Float64 => 6)

  for T in [Float32, Float64]
    Random.seed!(44)

    A = rand(T, n, n)
    A = A' * A + I
    b = rand(T, n)

    mulbyA!(y, x) = (y .= A * x)

    tol = sqrt(eps(T))
    gcrk = GeneralizedConjugateResidual(2, b, tol)
    
    x = rand(T, n)
    iters = linearsolve!(mulbyA!, x, b, gcrk)

    @test iters == expected_iters[T]
    @test norm(A * x - b, Inf) / norm(b, Inf) <= tol
   
    newtol = 1000tol
    settolerance!(gcrk, newtol)
    
    x = rand(T, n)
    linearsolve!(mulbyA!, x, b, gcrk)

    @test norm(A * x - b, Inf) / norm(b, Inf) <= newtol
    @test norm(A * x - b, Inf) / norm(b, Inf) >= tol

  end
end
