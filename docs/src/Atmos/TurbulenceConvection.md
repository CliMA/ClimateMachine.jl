# TurbulenceConvection

```@meta
CurrentModule = CLIMA.TurbulenceConvection
```

## Grids

```@docs
FiniteDifferenceGrids.Grid
FiniteDifferenceGrids.over_elems
FiniteDifferenceGrids.over_elems_real
FiniteDifferenceGrids.over_elems_ghost
FiniteDifferenceGrids.Zmin
FiniteDifferenceGrids.Zmax
FiniteDifferenceGrids.n_hat
FiniteDifferenceGrids.binary
FiniteDifferenceGrids.ghost_vec
FiniteDifferenceGrids.ghost_dual
FiniteDifferenceGrids.first_interior
FiniteDifferenceGrids.boundary
FiniteDifferenceGrids.second_interior
FiniteDifferenceGrids.first_ghost
FiniteDifferenceGrids.boundary_points
```

## Operators

```@docs
FiniteDifferenceGrids.grad
FiniteDifferenceGrids.∇_pos
FiniteDifferenceGrids.∇_neg
FiniteDifferenceGrids.∇_z_flux
FiniteDifferenceGrids.∇_z_centered
FiniteDifferenceGrids.∇_z_dual
FiniteDifferenceGrids.∇_z_upwind
FiniteDifferenceGrids.Δ_z
FiniteDifferenceGrids.Δ_z_dual
```

## State Vector

```@docs
StateVecs.StateVec
StateVecs.over_sub_domains
StateVecs.Cut
StateVecs.Dual
StateVecs.var_names
StateVecs.var_string
StateVecs.var_suffix
StateVecs.assign!
StateVecs.assign_real!
StateVecs.assign_ghost!
StateVecs.extrap!
StateVecs.extrap_0th_order!
StateVecs.compare
StateVecs.DomainIdx
StateVecs.subdomains
StateVecs.alldomains
StateVecs.eachdomain
StateVecs.allcombinations
StateVecs.DomainSubSet
StateVecs.get_param
StateVecs.DomainDecomp
```

## Examples

Several examples exist in the test directory.
