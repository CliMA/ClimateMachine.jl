"""
GridOperators

  Provides local finite-difference
  operator functions.

# Interface
 - [`∇_z`](@ref) gradient
 - [`Δ_z`](@ref) Laplacian
 - [`adv_upwind`](@ref) upwind advection
"""
module GridOperators

export ∇_z, Δ_z, adv_upwind
using Test, ..Grids

""" `∇_z(f, grid::Grid)`
computes the local derivative of field f """
∇_z(f, grid::Grid) = (f[1] + f[3] - 2*f[2])*grid.dzi

""" `Δ_z(f, grid::Grid)`
computes the local Laplacian of field f """
Δ_z(f, grid::Grid) = (f[3] - 2*f[2] + f[1])*grid.dzi2

""" `adv_upwind(ϕ, u, grid::Grid)`
Local upwind advection operator (stable at high Re, but diffusive) """
adv_upwind(ϕ, u, grid::Grid) = u[2] > 0 ? u[ 2 ]*(ϕ[2] - ϕ[1]) * grid.dzi :
                                          u[ 2 ]*(ϕ[3] - ϕ[2]) * grid.dzi

end