using Test
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver

using LinearAlgebra
using Random

@testset "LinearSolvers small system" begin
  n = 8
  Random.seed!(44)

  expected_iters = Dict(Float32 => 12, Float64 => 5)

  for T in [Float32, Float64]
    B = rand(T, n, n)
    A = B' * B
    
    xsol = rand(T, n)
    normalize!(xsol)

    b = A * xsol

    mulbyA!(y, x) = (y .= A * x)

    tol = eps(T)
    gcrk = GeneralizedConjugateResidual(3, b, tol)
    
    x = rand(T, n)
    iters = linearsolve!(mulbyA!, x, b, gcrk)

    @test iters == expected_iters[T]
    @test norm(A * x - b, Inf) <= 20tol
  end
end
