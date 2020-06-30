# TurbulenceClosures

```@meta
CurrentModule = ClimateMachine.TurbulenceClosures
```

```@docs
TurbulenceClosures
```

## Turbulence Closure Model Constructors

```@docs
TurbulenceClosureModel
ConstantViscosityWithDivergence
SmagorinskyLilly
Vreman
AnisoMinDiss
```

## Supporting Methods

```@docs
turbulence_tensors
init_aux_turbulence!
turbulence_nodal_update_auxiliary_state!
principal_invariants
symmetrize
norm2
strain_rate_magnitude
```
