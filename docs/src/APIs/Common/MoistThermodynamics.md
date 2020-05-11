# Moist Thermodynamics

```@meta
CurrentModule = ClimateMachine.MoistThermodynamics
```

## Thermodynamic State Constructors

```@docs
PhasePartition
PhasePartition_equil
ThermodynamicState
PhaseDry
PhaseEquil
PhaseNonEquil
TemperatureSHumEquil
LiquidIcePotTempSHumEquil
LiquidIcePotTempSHumNonEquil
LiquidIcePotTempSHumNonEquil_given_pressure
```

## Thermodynamic state methods
```@docs
air_density
air_pressure
air_temperature
air_temperature_from_liquid_ice_pottemp
cp_m
cv_m
dry_pottemp
exner
gas_constant_air
Ice
internal_energy
internal_energy_sat
latent_heat_fusion
latent_heat_sublim
latent_heat_vapor
Liquid
liquid_fraction
liquid_ice_pottemp
liquid_ice_pottemp_sat
gas_constants
saturation_adjustment
saturation_excess
q_vap_saturation
q_vap_saturation_generic
saturation_vapor_pressure
soundspeed_air
specific_volume
total_energy
virtual_pottemp
```
