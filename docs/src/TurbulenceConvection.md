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

## Grid operators

```@docs
GridOperators.∇_z
GridOperators.Δ_z
GridOperators.adv_upwind
```

## State vector

```@docs
StateVecs.StateVec
StateVecs.over_sub_domains
StateVecs.Cut
```

## Boundary conditions
```@docs
BoundaryConditions.Dirichlet!
BoundaryConditions.Neumann!
BoundaryConditions.Top
BoundaryConditions.Bottom
```

## State vector functions
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

