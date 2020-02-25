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

  methods = ((b, tol) -> GeneralizedConjugateResidual(2, b, rtol=tol),
             (b, tol) -> GeneralizedMinimalResidual(b, M=3, rtol=tol),
             (b, tol) -> GeneralizedMinimalResidual(b, M=n, rtol=tol)
            )

  expected_iters = (Dict(Float32 => 7, Float64 => 11),
                    Dict(Float32 => 5, Float64 => 17),
                    Dict(Float32 => 4, Float64 => 10)
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
    x0 = copy(x)
    iters = linearsolve!(mulbyA!, linearsolver, x, b; max_iters=Inf)

    @test iters == expected_iters[m][T]
    @test norm(A * x - b) / norm(A * x0 - b) <= tol

    # test for convergence in 0 iterations by
    # initializing with the exact solution
    x = A \ b
    iters = linearsolve!(mulbyA!, linearsolver, x, b; max_iters=Inf)
    @test iters == 0
    @test norm(A * x - b) <= 100eps(T)

    newtol = 1000tol
    settolerance!(linearsolver, newtol)

    x = @MVector rand(T, n)
    x0 = copy(x)
    linearsolve!(mulbyA!, linearsolver, x, b; max_iters=Inf)

    @test norm(A * x - b) / norm(A * x0 - b) <= newtol
    @test norm(A * x - b) / norm(A * x0 - b) >= tol

  end
end

@testset "LinearSolvers large full system" begin
  n = 1000

  methods = (
             (b, tol) -> GeneralizedMinimalResidual(b, M=15, rtol=tol),
             (b, tol) -> GeneralizedMinimalResidual(b, M=20, rtol=tol)
            )

  expected_iters = (
                    Dict(Float32 => (3,3), Float64 => (9, 8)),
                    Dict(Float32 => (3,3), Float64 => (9, 8))
                   )

  for (m, method) in enumerate(methods), T in [Float32, Float64]
    for (i, α) in enumerate(T[1e-2, 5e-3])
      Random.seed!(44)
      A = rand(T, 200, 1000)
      A = α * A' * A + I
      b = rand(T, n)

      mulbyA!(y, x) = (y .= A * x)

      tol = sqrt(eps(T))
      linearsolver = method(b, tol)

      x = rand(T, n)
      x0 = copy(x)
      iters = linearsolve!(mulbyA!, linearsolver, x, b; max_iters=Inf)

      @test iters == expected_iters[m][T][i]
      @test norm(A * x - b) / norm(A * x0 - b) <= tol

      newtol = 1000tol
      settolerance!(linearsolver, newtol)

      x = rand(T, n)
      x0 = copy(x)
      linearsolve!(mulbyA!, linearsolver, x, b; max_iters=Inf)

      @test norm(A * x - b) / norm(A * x0 - b) <= newtol
    end
  end
end
