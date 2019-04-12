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

``∂_t T = K ΔT + 1, T = 0 ∈ ∂Ω``

using Explicit Euler with `StateVec` and `Grid` constructs.

```@example Diffusion equation

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.GridOperators
using CLIMA.TurbulenceConvection.BoundaryConditions
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.StateVecFuncs

n_sd = 1 # number of sub-domains
K = 1.0 # diffusion coefficient
maxiter = 1000 # Explicit Euler iterations
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
plot_state(q, grid, "./", "T", :T) # for visualizing
nothing # hide
```
![](T.png)
