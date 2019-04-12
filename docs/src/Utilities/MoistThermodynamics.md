# MoistThermodynamics Module

The `MoistThermodynamics` module provides all thermodynamic functions needed for the atmosphere and functions shared across model components. The functions are general for a moist atmosphere that includes suspended cloud condensate in the working fluid; the special case of a dry atmosphere is obtained for zero specific humidities (or simply by omitting the optional specific humidity arguments in the functions that are needed for a dry atmosphere). The general formulation assumes that there are tracers for the total water specific humidity `q_tot`, the liquid specific humidity `q_liq`, and the ice specific humidity `q_ice` to characterize the thermodynamic state and composition of moist air.

There are several types of functions:

1. Equation of state (ideal gas law):
    * `air_pressure`
2. Specific gas constant and isobaric and isochoric specific heats of moist air:
    * `gas_constant_air`
    * `cp_m`
    * `cv_m`
3. Specific latent heats of vaporization, fusion, and sublimation:
    * `latent_heat_vapor`
    * `latent_heat_fusion`
    * `latent_heat_sublim`
4. Saturation vapor pressure and specific humidity over liquid and ice:
    * `sat_vapor_press_liquid`
    * `sat_vapor_press_ice`
    * `sat_shum`
5. Functions computing energies and inverting them to obtain temperatures
    * `total_energy`
    * `internal_energy`
    * `air_temperature`
6. Functions to compute temperatures and partitioning of water into phases in thermodynamic equilibrium (when Gibbs' phase rule implies that the entire thermodynamic state of moist air, including the liquid and ice specific humidities, can be calculated from the 3 thermodynamic state variables, such as energy, pressure, and total specific humidity)
    * `liquid_fraction_equil` (fraction of condensate that is liquid)
    * `saturation_adjustment` (compute temperature from energy, density, and total specific humidity)
7. Auxiliary functions for diagnostic purposes, e.g., other thermodynamic quantities
    * `liquid_ice_pottemp` (liquid-ice potential temperature)

A moist dynamical core that assumes equilibrium thermodynamics can be obtained from a dry dynamical core with total energy as a prognostic variable by including a tracer for the total specific humidity `q_tot`, using the functions, e.g., for the energies in the module, and computing the temperature `T` and the liquid and ice specific humidities `q_liq` and `q_ice` from the internal energy `e_int` by saturation adjustment:
```julia
    T = saturation_adjustment(e_int, ρ, q_tot);
    q_liq, q_ice = phase_partitioning_eq(T, ρ, q_tot);
```
here, `ρ` is the density of the moist air, and the internal energy `e_int = e_tot - e_kin - geopotential` is the total energy `e_tot` minus kinetic energy `e_kin` and potential energy `geopotential` (all energies per unit mass). No changes to the "right-hand sides" of the dynamical equations are needed for a moist dynamical core that supports clouds, as long as they do not precipitate. Additional source-sink terms arise from precipitation.

Schematically, the workflow in such a core would look as follows:
```julia

    # initialize
    geopotential = grav * z
    q_tot          = ...
    ρ            = ...

    (u, v, w)    = ...
    e_kin           = 0.5 * (u.^2 .+ v.^2 .+ w.^2)

    e_tot        = total_energy(e_kin, geopotential, T, q_tot)

    do timestep   # timestepping loop

      # advance dynamical variables by a timestep (temperature typically
      # appears in terms on the rhs, such as radiative transfer)
      advance(u, v, w, ρ, e_tot, q_tot)

      # compute internal energy from dynamic variables
      e_int = e_tot - 0.5 * (u.^2 .+ v.^2 .+ w.^2) - geopotential

      # compute temperature, pressure and condensate specific humidities,
      T = saturation_adjustment(e_int, ρ, q_tot);
      q_liq, q_ice = phase_partitioning_eq(T, ρ, q_tot);
      p = air_pressure(T, ρ, q_tot, q_liq, q_ice)

    end
```

For a dynamical core that additionally uses the liquid and ice specific humidities `q_liq` and `q_ice` as prognostic variables, and thus explicitly allows the presence of non-equilibrium phases such as supercooled water, the saturation adjustment in the above workflow is replaced by a direct calculation of temperature and pressure:
```julia
    T = air_temperature(e_int, q_tot, q_liq, q_ice)
    p = air_pressure(T, ρ, q_tot, q_liq, q_ice)
```

## Functions

```@meta
CurrentModule = CLIMA.MoistThermodynamics
```

```@docs
ThermodynamicState
PhaseEquil
PhaseNonEquil
LiquidIcePotTempSHumEquil
```

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
liquid_fraction_equil
liquid_fraction_nonequil
liquid_ice_pottemp
liquid_ice_pottemp_sat
moist_gas_constants
phase_partitioning_eq
saturation_adjustment
saturation_excess
saturation_shum
saturation_shum_generic
saturation_vapor_pressure
soundspeed_air
specific_volume
total_energy
virtual_pottemp
```





