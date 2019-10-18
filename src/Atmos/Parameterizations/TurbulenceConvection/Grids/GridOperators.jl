####    GridOperators

# Local finite-difference operator functions.

export advect
export grad, ∇_pos, ∇_neg

export ∇_z_flux, ∇_z_centered, ∇_z_dual, ∇_z_upwind
export Δ_z, Δ_z_dual

export advect_old
export UpwindAdvective,
       UpwindCollocated,
       OneSidedUp,
       OneSidedDn,
       CenteredUnstable

function advect(f, w, grid::Grid)
  @assert length(f)==3
  @assert length(w)==3
  if w[2] < 0
    return (f[3]-f[2])*grid.Δzi
  else
    return (f[2]-f[1])*grid.Δzi
  end
end

"""
    grad(f, grid::Grid{FT}) where FT

Computes local finite-difference gradient of `f`: `∇f`
"""
function grad(f, grid::Grid{FT}) where FT
  @assert length(f)==3 || length(f)==2
  if length(f)==2
    return (f[2]-f[1])*grid.Δzi
  else
    return (f[3]-f[1])*FT(0.5)*grid.Δzi
  end
end

"""
    ∇_pos(f, grid::Grid)

Computes a one-sided (up) local finite-difference gradient of `f`: `∇f`
"""
function ∇_pos(f, grid::Grid)
  @assert length(f)==3
  return (f[2]-f[1])*grid.Δzi
end

"""
    ∇_neg(f, grid::Grid)

Computes a one-sided (down) local finite-difference gradient of `f`: `∇f`
"""
function ∇_neg(f, grid::Grid)
  @assert length(f)==3
  return (f[3]-f[2])*grid.Δzi
end


"""
    ∇_z_flux(f, grid::Grid)

Computes the local derivative of field ``f``, which is
assumed to live on the dual grid: ``∇f``
"""
function ∇_z_flux(f, grid::Grid)
  @assert length(f)==2
  (f[2] - f[1])*grid.Δzi
end

"""
    ∇_z_centered(f, grid::Grid)

Computes the local derivative of field ``f``, which is
assumed to live on the dual grid: ``∇f``
"""
function ∇_z_centered(f, grid::Grid)
  @assert length(f)==3
  (f[3] - f[1])*grid.Δzi/2
end

"""
    ∇_z_dual(f, grid::Grid)

Computes the local derivative of field ``f``, which is
assumed to live on the dual grid: ``∇f``
"""
function ∇_z_dual(f, grid::Grid)
  @assert length(f)==3
  return [∇_z_flux(f[1:2], grid), ∇_z_flux(f[2:3], grid)]
end

"""
    ∇_z_upwind(ϕ, w, grid::Grid)

Local upwind gradient operator ``∇ϕ``. This
operator is stable but numerically diffusive.
"""
function ∇_z_upwind(ϕ, w, grid::Grid)
  @assert length(ϕ)==3
  @assert length(w)==3
  return w[2]>0 ? ∇_z_flux(ϕ[1:2], grid) : ∇_z_flux(ϕ[2:3], grid)
end

"""
    Δ_z(f, grid::Grid)

Computes the local Laplacian of field ``f``:
``∇ • (∇f)``
"""
function Δ_z(f, grid::Grid)
  @assert length(f)==3
  (f[3] - 2*f[2] + f[1])*grid.Δzi2
end

"""
    Δ_z(f, grid::Grid, K)

Computes the local Laplacian of field ``f``
with a variable coefficient:
``∇ • (K ∇f)``
"""
function Δ_z(f, grid::Grid, K)
  @assert length(f)==3
  K∇f_M = (K[1]+K[2])/2*(f[2]-f[1])*grid.Δzi
  K∇f_P = (K[2]+K[3])/2*(f[3]-f[2])*grid.Δzi
  return (K∇f_P - K∇f_M)*grid.Δzi
end

"""
    Δ_z_dual(f, grid::Grid, K)

Computes the local Laplacian of field ``f``
with a variable coefficient:
``∇ • (K ∇f)``
"""
function Δ_z_dual(f, grid::Grid, K)
  @assert length(f)==3
  @assert length(K)==2
  K∇f_M = K[1]*(f[2]-f[1])*grid.Δzi
  K∇f_P = K[2]*(f[3]-f[2])*grid.Δzi
  return (K∇f_P - K∇f_M)*grid.Δzi
end

abstract type AdvectForm end
struct UpwindAdvective <: AdvectForm end

abstract type ConservativeForm end
struct UpwindCollocated <: ConservativeForm end
struct OneSidedUp <: ConservativeForm end
struct OneSidedDn <: ConservativeForm end
struct CenteredUnstable <: ConservativeForm end

"""
    advect(ϕ, w, grid::Grid, ::AdvectForm, Δt)

Advection operator, which computes

 - ``w • ∇ϕ`` given `AdvectiveForm`
 - ``∇ • (wϕ)`` given `ConservativeForm`

which is dispatched by `AdvectForm`.
"""
function advect_old(ϕ, ϕ_dual, w, w_dual, grid::Grid, ::UpwindAdvective, Δt)
  @assert length(ϕ)==3
  @assert length(ϕ_dual)==2
  @assert length(w)==3
  @assert length(w_dual)==2
  if w[2] > 0
    return w[ 2 ]*(ϕ[2] - ϕ[1]) * grid.Δzi
  else
    return w[ 2 ]*(ϕ[3] - ϕ[2]) * grid.Δzi
  end
end
function advect_old(ϕ, ϕ_dual, w, w_dual, grid::Grid, ::UpwindCollocated, Δt)
  @assert length(ϕ)==3
  @assert length(ϕ_dual)==2
  @assert length(w)==3
  @assert length(w_dual)==2
  if sum(w)/length(w) > 0
    return (w[2]*ϕ[2] - w[1]*ϕ[1]) * grid.Δzi
  else
    return (w[3]*ϕ[3] - w[2]*ϕ[2]) * grid.Δzi
  end
end
function advect_old(ϕ, ϕ_dual, w, w_dual, grid::Grid, ::OneSidedUp, Δt)
  @assert length(ϕ)==3
  @assert length(ϕ_dual)==2
  @assert length(w)==3
  @assert length(w_dual)==2
  return (w[2]*ϕ[2] - w[1]*ϕ[1]) * grid.Δzi
end
function advect_old(ϕ, ϕ_dual, w, w_dual, grid::Grid, ::OneSidedDn, Δt)
  @assert length(ϕ)==3
  @assert length(ϕ_dual)==2
  @assert length(w)==3
  @assert length(w_dual)==2
  return (w[3]*ϕ[3] - w[2]*ϕ[2]) * grid.Δzi
end
function advect_old(ϕ, ϕ_dual, w, w_dual, grid::Grid, ::CenteredUnstable, Δt)
  @assert length(ϕ)==3
  @assert length(ϕ_dual)==2
  @assert length(w)==3
  @assert length(w_dual)==2
  return (w_dual[2]*ϕ_dual[2]-w_dual[1]*ϕ_dual[1])*grid.Δzi
end
