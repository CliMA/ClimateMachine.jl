using Test
using CLIMA
using CLIMA.LinearSolvers
using CLIMA.GeneralizedConjugateResidualSolver

using LinearAlgebra

@testset "LinearSolvers small system" begin
  n = 8

  expected_iters = Dict(Float32 => 21, Float64 => 11)

  for T in [Float32, Float64]
    B = T[mod(i + j, 3) + mod(2i - j, 5) for j = 1:n, i = 1:n]
    A = B' * B

    A ./= (maximum(A))
    
    xsol = T[i ^ 2 for i = 1:n]
    normalize!(xsol)

    b = A * xsol

    mulbyA!(y, x) = (y .= A * x)

    tol = eps(T)
    gcrk = GeneralizedConjugateResidual(3, b, tol)
    
    x = ones(T, n)
    iters = linearsolve!(mulbyA!, x, b, gcrk)

    @test iters == expected_iters[T]
    @test norm(A * x - b, Inf) <= 20tol
   
    newtol = 1000tol
    settolerance!(gcrk, newtol)
    
    x = ones(T, n)
    linearsolve!(mulbyA!, x, b, gcrk)

    @test norm(A * x - b, Inf) <= 20newtol
    @test norm(A * x - b, Inf) >= tol

  end
end
