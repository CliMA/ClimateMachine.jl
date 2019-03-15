using Test

using LinearAlgebra

using Solvers: TriDiagSolverFuncs
using Grids
using Fields

@testset "TriDiagSolverFuncs" begin
  N = 2:10
  tol = 0.00000001
  for n in N
    dl = rand(n-1)
    du = rand(n-1)
    d = rand(n);

    A = Array(Tridiagonal(dl, d, du))
    b = rand(length(d))

    x_correct = inv(A)*b

    xtemp = zeros(n)
    gamma = zeros(n-1)
    beta = zeros(n)
    x_TDMA = zeros(n)

    TriDiagSolverFuncs.solve_tridiag!(x_TDMA, b, dl, d, du, n, xtemp, gamma, beta)
    @test all([abs(x-y)<tol for (x, y) in zip(x_correct, x_TDMA)])

    TriDiagSolverFuncs.init_beta_gamma!(beta, gamma, dl, d, du, n)
    TriDiagSolverFuncs.solve_tridiag_stored!(x_TDMA, b, dl, beta, gamma, n, xtemp)

    @test all([abs(x-y)<tol for (x, y) in zip(x_correct, x_TDMA)])
  end
end

