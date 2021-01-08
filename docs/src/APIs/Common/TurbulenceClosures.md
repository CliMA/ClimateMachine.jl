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
hyperdiff_enthalpy_and_momentum_flux
hyperdiff_momentum_flux
WithDivergence
WithoutDivergence
ConstantViscosity
ConstantDynamicViscosity
ConstantKinematicViscosity
SmagorinskyLilly
Vreman
AnisoMinDiss
HyperDiffusion
NoHyperDiffusion
Biharmonic
ViscousSponge
NoViscousSponge
UpperAtmosSponge
```

## Supporting Methods

```@docs
turbulence_tensors
init_aux_turbulence!
principal_invariants
symmetrize
norm2
strain_rate_magnitude
```
