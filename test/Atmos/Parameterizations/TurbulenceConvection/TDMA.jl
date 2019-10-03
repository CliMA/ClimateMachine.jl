using Test

using LinearAlgebra

using CLIMA.TurbulenceConvection.TriDiagSolvers
TDMA = TriDiagSolvers

@testset "TriDiagSolvers" begin
  N = 2:10
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

    solve_tridiag!(x_TDMA, b, dl, d, du, n, xtemp, gamma, beta)
    tol = eps(Float32)

    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    # @show eps(Float64)
    # @show "Full", err
    if !all([x<tol for x in err])
        @show tol
        @show "Full", err
    end
    @assert all([x<tol for x in err])

    init_β_γ!(beta, gamma, dl, d, du, n)
    solve_tridiag_stored!(x_TDMA, b, dl, beta, gamma, n, xtemp)

    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    # @show "Stored", err
    @assert all([x<tol for x in err])

    dl_mod = zeros(n)
    dl_mod[2:end] = dl
    du_mod = zeros(n)
    du_mod[1:end-1] = du
    TDMA.solve_tridiag_old(n, b, dl_mod, d, du_mod)
    x_TDMA = b
    err = [abs(x-y) for (x, y) in zip(x_correct, x_TDMA)]
    # @show "old", err
    @assert all([x<tol for x in err])

  end
end

