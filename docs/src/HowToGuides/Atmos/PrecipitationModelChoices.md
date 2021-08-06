# Precipitation model choices in Atmos.jl

The precipitation model in `Atmos.jl` describes the behavior
  of precipitating water in the atmosphere (i.e. rain and snow).
There are two options available: `NoPrecipitation` and `RainModel`.

## NoPrecipitation

The `NoPrecipitation` model assumes there is no precipitating water present
  in any form in the simulation.
It does not add any precipitation related variables to state variables.
This model can be used (i) without defining any precipitation related source terms,
  (ii) or using the [0-moment microphysics scheme](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_0M/).
In the first case `ρ q_tot` (total water specific humidity) is not removed
  during the simulation.
In the second case `ρ q_tot` is removed if it exceeds a threshold.


## RainModel

The `RainModel` assumes that precipitating water is present but only in the
  form of rain (liquid-phase precipitation).
It adds `ρ q_rai` (air density times total rain water specific humidity)
  to state variables.
It uses a subset of source terms from the [1-moment microphysics scheme](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/)
  that parameterize processes relevant to liquid-phase clouds
  (autoconverion, accretion, and rain evaporation).

## RainSnowModel

The `RainSnowModel` assumes that precipitating water is present in the
  form of rain (liquid-phase precipitation) and
  snow (ice-phase precipitation).
It adds `ρ q_rai` (air density times total rain water specific humidity)
  and `ρ q_sno` (air density times total snow water specific humidity)
  to state variables.
It uses source terms from the [1-moment microphysics scheme](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/).
