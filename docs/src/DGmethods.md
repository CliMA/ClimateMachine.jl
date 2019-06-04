# ODESolvers

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

### Functions
```@docs
SpaceMethods.odefun!
```

## `DGBalanceLawDiscretizations`

```@docs
DGBalanceLawDiscretizations
```

### Types
```@docs
DGBalanceLawDiscretizations.DGBalanceLaw
```
### Constructor
```@docs
DGBalanceLawDiscretizations.DGBalanceLaw(;)
```

### Functions
```@docs
DGBalanceLawDiscretizations.grad_auxiliary_state!
DGBalanceLawDiscretizations.indefinite_stack_integral!
DGBalanceLawDiscretizations.dof_iteration!
```

### Pirated Type Functions
```@docs
CLIMA.MPIStateArrays.MPIStateArray(::DGBalanceLawDiscretizations.DGBalanceLaw)
CLIMA.MPIStateArrays.MPIStateArray(::DGBalanceLawDiscretizations.DGBalanceLaw, ::Function)
CLIMA.SpaceMethods.odefun!(::DGBalanceLawDiscretizations.DGBalanceLaw, dQ, Q, t)
```

### Kernels
```@docs
DGBalanceLawDiscretizations.volumerhs!
DGBalanceLawDiscretizations.facerhs!
DGBalanceLawDiscretizations.volumeviscterms!
DGBalanceLawDiscretizations.faceviscterms!
DGBalanceLawDiscretizations.initauxstate!
DGBalanceLawDiscretizations.initauxstate!
DGBalanceLawDiscretizations.elem_grad_field!
DGBalanceLawDiscretizations.knl_dof_iteration!
DGBalanceLawDiscretizations.knl_indefinite_stack_integral!
```

## `DGBalanceLawDiscretizations.NumericalFluxes`

```@docs
DGBalanceLawDiscretizations.NumericalFluxes.rusanov!
DGBalanceLawDiscretizations.rusanov_boundary_flux!
```
