# CLIMA Utilities
Functions shared across model components, e.g., for thermodynamics.

## MoistThermodynamics Module

The `MoistThermodynamics` module provides all thermodynamic functions needed for the atmosphere and functions shared across model components. The functions are general for a moist atmosphere that includes suspended cloud condensate in the working fluid; the special case of a dry atmosphere is obtained for zero specific humidities (or simply by omitting the optional specific humidity arguments in the functions that are needed for a dry atmosphere). The general formulation assumes that there are tracers for the total water specific humidity `q_t`, the liquid specific humidity `q_l`, and the ice specific humidity `q_i` to characterize the thermodynamic state and composition of moist air.

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
    * `liquid_fraction` (fraction of condensate that is liquid)
    * `saturation_adjustment` (compute temperature, density, and condensate specific humidities from energy, density, and total specific humidity)
7. Auxiliary functions for diagnostic purposes, e.g., other thermodynamic quantities
    * `liquid_ice_pottemp` (liquid-ice potential temperature)

A moist dynamical core that assumes equilibrium thermodynamics can be obtained from a dry dynamical core with total energy as a prognostic variable by including a tracer for the total specific humidity `q_t`, using the functions, e.g., for the energies in the module, and computing the temperature `T` and the liquid and ice specific humidities `q_l` and `q_i` from the internal energy `E_int` by saturation adjustment:
```julia
    T, p, q_l, q_i   = saturation_adjustment(E_int, ρ, q_t, T_init);
```
here, `ρ` is the density of the moist air, `T_init` is an initial temperature guess for the saturation adjustment iterations, and the internal energy `E_int = E_tot - KE - geopotential` is the total energy `E_tot` minus kinetic energy `KE` and potential energy `geopotential` (all energies per unit mass). No changes to the "right-hand sides" of the dynamical equations are needed for a moist dynamical core that supports clouds, as long as they do not precipitate. Additional source-sink terms arise from precipitation.

Schematically, the workflow in such a core would look as follows:
```julia

    # initialize
    geopotential = grav * z
    T_prev       = ...
    q_t          = ...
    ρ            = ...

    (u, v, w)    = ...
    KE           = 0.5 * (u.^2 .+ v.^2 .+ w.^2)

    E_tot        = total_energy(KE, geopotential, T, q_t)

    do timestep   # timestepping loop

      # advance dynamical variables by a timestep (temperature typically
      # appears in terms on the rhs, such as radiative transfer)
      advance(u, v, w, ρ, E_tot, q_t)  

      # compute internal energy from dynamic variables
      E_int = E_tot - 0.5 * (u.^2 .+ v.^2 .+ w.^2) - geopotential

      # compute temperature, pressure and condensate specific humidities,
      # using T_prev as initial condition for iterations
      T, p, q_l, q_i = saturation_adjustment(E_int, ρ, q_t, T_prev);

      # update temperature for next timestep
      T_prev = T;  
    end
```

For a dynamical core that additionally uses the liquid and ice specific humidities `q_l` and `q_i` as prognostic variables, and thus explicitly allows the presence of non-equilibrium phases such as supercooled water, the saturation adjustment in the above workflow is replaced by a direct calculation of temperature and pressure:
```julia
    T = air_temperature(E_int, q_t, q_l, q_i)
    p = air_pressure(T, ρ, q_t, q_l, q_i)
```
