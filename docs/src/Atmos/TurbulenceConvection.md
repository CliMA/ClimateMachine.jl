# `TurbulenceConvection`

```@meta
CurrentModule = CLIMA.TurbulenceConvection
```

## Grids

```@docs
Grids.Grid
Grids.over_elems
Grids.over_elems_real
Grids.over_elems_ghost
Grids.first_elem_above_surface
Grids.get_z
```

## Grid Operators

```@docs
GridOperators.∇_z
GridOperators.Δ_z
GridOperators.adv_upwind
```

## State Vector

```@docs
StateVecs.StateVec
StateVecs.over_sub_domains
StateVecs.Cut
```

## Boundary Conditions
```@docs
BoundaryConditions.Dirichlet!
BoundaryConditions.Neumann!
BoundaryConditions.Top
BoundaryConditions.Bottom
```

## State Vector Functions
```@docs
StateVecFuncs.first_elem_above_surface_val
StateVecFuncs.surface_val
StateVecFuncs.extrap!
StateVecFuncs.assign_ghost!
StateVecFuncs.domain_average!
StateVecFuncs.distribute!
StateVecFuncs.total_covariance!
StateVecFuncs.export_state
```

## Solving a diffusion equation

Here, we solve a simple diffusion equation

``∂_t T = K ΔT + 1, \qquad T = 0 ∈ ∂Ω``

using Explicit Euler with `StateVec` and `Grid` constructs.

```@example Diffusion equation

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.GridOperators
using CLIMA.TurbulenceConvection.BoundaryConditions
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.StateVecFuncs
using Plots

n_sd = 1 # number of sub-domains
K = 1.0 # diffusion coefficient
maxiter = 1000 # time-step iterations
Δt = 0.001 # time step

grid = Grid(0.0, 1.0, 10)
q = StateVec(((:T, n_sd),), grid)
rhs = deepcopy(q)

for i in 1:maxiter
  for k in over_elems_real(grid)
    rhs[:T, k] = K*Δ_z(q[:T, Cut(k)], grid) + 1
  end
  for k in over_elems(grid)
    q[:T, k] += Δt*rhs[:T, k]
  end
  Dirichlet!(q, :T, 0.0, grid, Top())
  Dirichlet!(q, :T, 0.0, grid, Bottom())
end
plot_state(q, grid, "./", "T.svg", :T) # for visualizing
nothing # hide
```
![](T.svg)

## Solving a variable coefficient diffusion equation

Here, we solve a variable coefficient diffusion equation

``∂_t T = ∇ • (K(z)∇T) + 1, \qquad T = 0 ∈ ∂Ω``

``K(z) = 1 - .9 \times H(z-.5), \qquad H = \text{heaviside}``

using Explicit Euler.

```@example Variable coefficient diffusion equation

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.GridOperators
using CLIMA.TurbulenceConvection.BoundaryConditions
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.StateVecFuncs
using Plots

n_sd = 1 # number of sub-domains
maxiter = 10000 # time-step iterations
Δt = 0.001 # time step

grid = Grid(0.0, 1.0, 10)
unknowns = ( (:T, n_sd), )
vars = ( (:ΔT, n_sd), (:K_thermal, n_sd) )
q = StateVec(unknowns, grid)
tmp = StateVec(vars, grid)
rhs = deepcopy(q)

cond_thermal(z) = z > .5 ? 1 : .1
for i in 1:maxiter
  for k in over_elems_real(grid)
    tmp[:K_thermal, k] = cond_thermal(get_z(grid, k))
    tmp[:ΔT, k] = Δ_z(q[:T, Cut(k)], grid, tmp[:K_thermal, Cut(k)])
    rhs[:T, k] = tmp[:ΔT, k] + 1
  end
  for k in over_elems(grid)
    q[:T, k] += Δt*rhs[:T, k]
  end
  Dirichlet!(q, :T, 0.0, grid, Top())
  Dirichlet!(q, :T, 0.0, grid, Bottom())
end
plot_state(q, grid, "./", "T_varK.svg", :T) # for visualizing
nothing # hide
```
![](T_varK.svg)

## Solving a linear advection equation

Here, we solve a linear advection equation

``∂_t u + c∇u = 0, \qquad u = 0 ∈ ∂Ω``

``u(t=0) = Gaussian(σ, μ)``

using Explicit Euler method.

```@example Diffusion equation

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.GridOperators
using CLIMA.TurbulenceConvection.BoundaryConditions
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.StateVecFuncs
using Plots

n_sd = 1 # number of sub-domains
maxiter = 400 # time-step iterations
Δt = 0.0005 # time step

grid = Grid(0.0, 1.0, 200)
unknowns = ( (:u, n_sd), )
vars = ( (:u_initial, n_sd), )
q = StateVec(unknowns, grid)
tmp = StateVec(vars, grid)
rhs = deepcopy(q)

σ, μ, c = .05, 0.3, 1.0
T = maxiter*Δt
ic(z) = 1/(σ*sqrt(2*π))*exp(-1/2*((z-μ)/σ)^2)
for k in over_elems_real(grid)
  tmp[:u_initial, k] = ic(get_z(grid, k))
  q[:u, k] = tmp[:u_initial, k]
end
plot_state(tmp, grid, "./", "u_initial.svg", :u_initial) # for visualizing
for i in 1:maxiter
  for k in over_elems_real(grid)
    rhs[:u, k] = - adv_upwind(q[:u, Cut(k)], c .* [1,1,1], grid)
  end
  for k in over_elems(grid)
    q[:u, k] += Δt*rhs[:u, k]
  end
  Dirichlet!(q, :u, 0.0, grid, Top())
  Dirichlet!(q, :u, 0.0, grid, Bottom())
end
plot_state(q, grid, "./", "u_final.svg", :u) # for visualizing
nothing # hide
```
![](u_initial.svg)
![](u_final.svg)
