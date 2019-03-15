module TriDiagSolvers

using ..TriDiagSolverFuncs
using ..TriDiags

using Grids
using Fields

"""
Solves, using a tri-diagonal system:

  ∂_t x - K ∂_zz x = b

  Consider a θ-implicit diffusion term:

  x^{n+1} - x^n - Δt θ K ∂_zz x^{n+1} = Δt b^n + Δt (1-θ) K ∂_zz x^n
  x^{n+1} - Δt θ K ∂_zz x^{n+1} = x^n + Δt b^n + Δt (1-θ) K ∂_zz x^n
  (I - Δt θ K ∂_zz) x^{n+1} = RHS^n = x^n + Δt b^n + Δt (1-θ) K ∂_zz x^n

  To generalize this, let's write this as:

  (c_primary1(z) + c_primary2(z)*c_dual(z) ∂_zz) x^{n+1} = RHS^n

To set c_1 and c_2, call init_coefficients!.

"""
struct TriDiagSolver{T}
  tri_diag::TriDiag
  xtemp::Vector{T}
  gamma::Vector{T}
  beta::Vector{T}
end

function TriDiagSolver(x::Field, grid::Grid, T)
  tri_diag = TriDiag(x.N - 2*grid.gw)
  xtemp = Vector{T}([0 for _ in 1:x.N - 2*grid.gw   ])
  gamma = Vector{T}([0 for _ in 1:x.N - 2*grid.gw -1])
  beta  = Vector{T}([0 for _ in 1:x.N - 2*grid.gw   ])
  return TriDiagSolver(dl, dd, du, xtemp, gamma, beta)
end

"""
Set coefficients c_1 and c_2 in the equation:
  (c_1(z) + c_2(z) K_dual(z)) x = b
where K_dual lives on the dual grid, compared to x.
"""
function init_coefficients!(tri_diag_solver::TriDiagSolver, # updated
                            grid::Grid,
                            x::Field,
                            c_1::Field,
                            c_2::Field,
                            K_dual::Field
                            )
  for k in over_elems_real(x, grid)
    tri_diag_solver.tri_diag.L[k-grid.gw+1] =          c_2[k]* K_dual[k-1]
    tri_diag_solver.tri_diag.D[k-grid.gw+1] = c_1[k] + c_2[k]*(K_dual[k-1] + K_dual[k])
    tri_diag_solver.tri_diag.U[k-grid.gw+1] =          c_2[k]*               K_dual[k]
  end
end

function init_tridiag_diffusion!(tri_diag_solver::TriDiagSolver, # updated
                                 x::Field,
                                 grid::Grid,
                                 Δt,
                                 θ,
                                 rho_ae_K_m::Field,
                                 rho::Field,
                                 a_env::Field)

  c_1 = Fields.init_temp(x)
  assign!(c_1, 1.0)

  c_2 = Fields.init_temp(x)
  assign!(c_2, -1.0)
  multiply!(c_2, θ*Δt*grid.dzi2)
  divide!(c_2, rho)
  divide!(c_2, a_env)

  K_dual = Fields.init_dual(x)
  assign!(K_dual, rho_ae_K_m)

  init_coefficients!(tri_diag_solver, grid, x, c_1, c_2, K_dual)
end

function solve!(x::Field, b::Field, grid::Grid, tri_diag_solver::TriDiagSolver)
  TriDiagSolverFuncs.solve_tridiag!(x.val[1+grid.gw:end-grid.gw],
                                    b.val[1+grid.gw:end-grid.gw],
                                    tri_diag_solver.tri_diag.L,
                                    tri_diag_solver.tri_diag.D,
                                    tri_diag_solver.tri_diag.U,
                                    x.N-2*grid.gw,
                                    tri_diag_solver.xtemp,
                                    tri_diag_solver.gamma,
                                    tri_diag_solver.beta)
end

end