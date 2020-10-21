# Thermodynamics

```@meta
CurrentModule = ClimateMachine.Thermodynamics
```
```@docs
Thermodynamics
```

## Thermodynamic State Constructors

```@docs
PhasePartition
PhasePartition_equil
ThermodynamicState
PhaseDry
PhaseDry_pT
PhaseDry_pθ
PhaseDry_ρT
PhaseEquil
PhaseEquil_ρTq
PhaseEquil_pTq
PhaseEquil_pθq
PhaseEquil_ρθq
PhaseNonEquil
PhaseNonEquil_ρTq
PhaseNonEquil_ρθq
PhaseNonEquil_pθq
```

## Thermodynamic state methods

```@docs
air_density
air_pressure
air_temperature
condensate
cp_m
cv_m
dry_pottemp
exner
gas_constant_air
gas_constants
has_condensate
Ice
ice_specific_humidity
internal_energy
internal_energy_sat
latent_heat_fusion
latent_heat_liq_ice
latent_heat_sublim
latent_heat_vapor
Liquid
liquid_fraction
liquid_ice_pottemp
liquid_ice_pottemp_sat
liquid_specific_humidity
moist_static_energy
q_vap_saturation
q_vap_saturation_liquid
q_vap_saturation_ice
q_vap_saturation_generic
relative_humidity
saturated
saturation_adjustment
saturation_excess
saturation_vapor_pressure
soundspeed_air
specific_enthalpy
specific_volume
supersaturation
total_energy
total_specific_enthalpy
total_specific_humidity
vapor_specific_humidity
virtual_pottemp
virtual_temperature
```

## Dispatch types

```@docs
DryAdiabaticProcess
```
