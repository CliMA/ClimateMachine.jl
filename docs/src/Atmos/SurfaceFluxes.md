# `SurfaceFluxes`

```@meta
CurrentModule = CLIMA.SurfaceFluxes
```

This module provides a means to compute surface fluxes given several variables, described in [`surface_conditions`](@ref).

## Interface
  - [`surface_conditions`](@ref) computes
    - Monin-Obukhov length
    - Potential temperature flux (if not given) using Monin-Obukhov theory
    - transport fluxes using Monin-Obukhov theory
    - friction velocity/temperature scale/tracer scales
    - exchange coefficients

## API

```@docs
surface_conditions
```

## References

- Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the Monin-Obukhov
  Similarity for Finite Volume Models." Journal of Advances in Modeling Earth Systems
  10.12 (2018): 3159-3175.
  doi: [10.1029/2018MS001534](https://doi.org/10.1029/2018MS001534)

- Businger, Joost A., et al. "Flux-profile relationships in the atmospheric surface
  layer." Journal of the atmospheric Sciences 28.2 (1971): 181-189.
  doi: [10.1175/1520-0469(1971)028<0181:FPRITA>2.0.CO;2](https://doi.org/10.1175/1520-0469(1971)028<0181:FPRITA>2.0.CO;2)

- Byun, Daewon W. "On the analytical solutions of flux-profile relationships for the
  atmospheric surface layer." Journal of Applied Meteorology 29.7 (1990): 652-657.
  doi: [10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2](http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2)

- Wyngaard, John C. "Modeling the planetary boundary layer-Extension to the stable case."
  Boundary-Layer Meteorology 9.4 (1975): 441-460.
  doi: [10.1007/BF00223393](https://doi.org/10.1007/BF00223393)
