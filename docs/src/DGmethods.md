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
DGBalanceLawDiscretizations.writevtk
DGBalanceLawDiscretizations.writevtk_helper
DGBalanceLawDiscretizations.grad_auxiliary_state!
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
DGBalanceLawDiscretizations.initauxstate!
DGBalanceLawDiscretizations.elem_grad_field!
```

## `DGBalanceLawDiscretizations.NumericalFluxes`

```@docs
DGBalanceLawDiscretizations.NumericalFluxes.rusanov!
```
