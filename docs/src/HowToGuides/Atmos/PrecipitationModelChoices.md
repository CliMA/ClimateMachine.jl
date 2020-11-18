# Precipitation model choices in Atmos.jl

The precipitation model in `Atmos.jl` describes the behavior
  of precipitating water in the atmosphere (i.e. rain and snow).
There are two options available: `NoPrecipitation` and `Rain`.

## NoPrecipitation

The `NoPrecipitation` model assumes there is no precipitating water present
  in any form in the simulation.
It does not add any precipitation related variables to state variables.
This model can be used (i) without defining any precipitation related source terms,
  (ii) or using the [0-moment microphysics scheme](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/Microphysics_0M/).
In the first case `ρ q_tot` (total water specific humidity) is not removed
  during the simulation.
In the second case `ρ q_tot` is removed if it exceeds a threshold.


## Rain

The `Rain` model assumes that precipitating water is present but only in the
  form of rain (liquid-phase precipitation).
It adds `ρ q_rai` (air density times total rain water specific humidity)
  to state variables.
It uses a subset of source terms from the [1-moment microphysics scheme](https://clima.github.io/ClimateMachine.jl/latest/Theory/Atmos/Microphysics/)
  that parameterize processes relevant to liquid-phase clouds
  (autoconverion, accretion, and rain evaporation).

## RainSnow

Soon!

