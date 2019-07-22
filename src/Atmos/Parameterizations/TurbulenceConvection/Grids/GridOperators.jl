"""
    GridOperators

Provides local finite-difference
operator functions.
"""
module GridOperators

export ∇_z, Δ_z, adv_upwind, adv_upwind_conservative, upwind_conservative
using ..Grids

"""
    ∇_z(f, grid::Grid)

Computes the local derivative of field ``f``:
``∇f``
"""
∇_z(f, grid::Grid) = (f[3] - f[1])/(2*grid.dz)

"""
    Δ_z(f, grid::Grid)

Computes the local Laplacian of field ``f``:
``∇ • (∇f)``
"""
Δ_z(f, grid::Grid) = (f[3] - 2*f[2] + f[1])*grid.dzi2

"""
    Δ_z(f, grid::Grid, K)

Computes the local Laplacian of field ``f``
with a variable coefficient:
``∇ • (K ∇f)``
"""
function Δ_z(f, grid::Grid, K)
  K∇f_left  = (K[1]+K[2])/2*(f[2]-f[1])*grid.dzi
  K∇f_right = (K[2]+K[3])/2*(f[3]-f[2])*grid.dzi
  return (K∇f_right - K∇f_left)*grid.dzi
end

"""
    adv_upwind(ϕ, w, grid::Grid)

Local upwind advection operator ``w • ∇ϕ``. This
operator is stable but numerically diffusive.
"""
function adv_upwind(ϕ, w, grid::Grid)
  if w[2] > 0
    return w[ 2 ]*(ϕ[2] - ϕ[1]) * grid.dzi
  else
    return w[ 2 ]*(ϕ[3] - ϕ[2]) * grid.dzi
  end
end

"""
    adv_upwind_conservative(ϕ, w, grid::Grid)

Local upwind advection operator ``∇ • (wϕ)``. This
operator is stable but numerically diffusive.
"""
function adv_upwind_conservative(ϕ, w, grid::Grid)
  w_dual = [(w[1]+w[2])/2,
            (w[2]+w[3])/2]
  g_dual = [w_dual[1] > 0 ? ϕ[1] : ϕ[2],
            w_dual[2] > 0 ? ϕ[2] : ϕ[3]]
  return (w_dual[2]*g_dual[2] - w_dual[1]*g_dual[1]) * grid.dzi
end

"""
    upwind_conservative(ϕ, w, grid::Grid)

Local upwind gradient operator ``∇ϕ``. This
operator is stable but numerically diffusive.
"""
function upwind_conservative(ϕ, w, grid::Grid)
  w_dual = [(w[1]+w[2])/2,
            (w[2]+w[3])/2]
  g_dual = [w_dual[1] > 0 ? ϕ[1] : ϕ[2],
            w_dual[2] > 0 ? ϕ[2] : ϕ[3]]
  return (g_dual[2] - g_dual[1]) * grid.dzi
end

end
