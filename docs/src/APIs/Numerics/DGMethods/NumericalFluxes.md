# Numerical Fluxes

```@meta
CurrentModule = ClimateMachine.DGMethods.NumericalFluxes
```

## Methods for applying boundary conditions

```@docs
numerical_boundary_flux_first_order!
numerical_boundary_flux_second_order!
numerical_boundary_flux_gradient!
numerical_boundary_flux_divergence!
numerical_boundary_flux_higher_order!
```

## Types

```@docs
NumericalFluxGradient
RusanovNumericalFlux
RoeNumericalFlux
HLLCNumericalFlux
RoeNumericalFluxMoist
NumericalFluxFirstOrder
NumericalFluxSecondOrder
CentralNumericalFluxSecondOrder
CentralNumericalFluxFirstOrder
CentralNumericalFluxGradient
LMARSNumericalFlux
```
