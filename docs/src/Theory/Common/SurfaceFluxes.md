# Surface Fluxes

```@meta
CurrentModule = ClimateMachine.SurfaceFluxes
```

This module provides a means to compute surface fluxes given several variables, described in [`surface_conditions`](@ref SurfaceFluxes.surface_conditions).

## Interface
  - [`surface_conditions`](@ref SurfaceFluxes.surface_conditions) computes
    - Monin-Obukhov length
    - Potential temperature flux (if not given) using Monin-Obukhov theory
    - transport fluxes using Monin-Obukhov theory
    - friction velocity/temperature scale/tracer scales
    - exchange coefficients

## References

- [Businger1971](@cite)
- [Nishizawa2018](@cite)
- [Byun1990](@cite)
- [Wyngaard1975](@cite)
