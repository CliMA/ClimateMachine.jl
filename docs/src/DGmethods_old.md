# DGmethods_old

```@meta
CurrentModule = CLIMA
```

## `SpaceMethods`

Set of abstract types for spatial discretizations

### Types
```@docs
SpaceMethods.AbstractSpaceMethod
SpaceMethods.AbstractDGMethod
```

## `DGBalanceLawDiscretizations`

```@docs
DGBalanceLawDiscretizations
```

### Types/Constructors
```@docs
DGBalanceLawDiscretizations.DGBalanceLaw
```

### Functions
```@docs
DGBalanceLawDiscretizations.grad_auxiliary_state!
DGBalanceLawDiscretizations.indefinite_stack_integral!
DGBalanceLawDiscretizations.reverse_indefinite_stack_integral!
DGBalanceLawDiscretizations.dof_iteration!
```

### Pirated Type Functions
```@docs
CLIMA.MPIStateArrays.MPIStateArray(::DGBalanceLawDiscretizations.DGBalanceLaw)
CLIMA.MPIStateArrays.MPIStateArray(::DGBalanceLawDiscretizations.DGBalanceLaw, ::Function)
CLIMA.SpaceMethods.odefun!
```

### Kernels
```@docs
DGBalanceLawDiscretizations.volumerhs!
DGBalanceLawDiscretizations.facerhs!
DGBalanceLawDiscretizations.initauxstate!
DGBalanceLawDiscretizations.elem_grad_field!
DGBalanceLawDiscretizations.knl_dof_iteration!
DGBalanceLawDiscretizations.knl_indefinite_stack_integral!
```

## Numerical Fluxes

```@docs
DGBalanceLawDiscretizations.NumericalFluxes.rusanov!
DGBalanceLawDiscretizations.NumericalFluxes.rusanov_boundary_flux!
```
