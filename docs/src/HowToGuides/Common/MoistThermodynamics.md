# [MoistThermodynamics Module](@id MoistThermodynamics-docs)

MoistThermodynamics.jl provides all thermodynamic functions needed for the
atmosphere and functions shared across model components. The functions are
general for a moist atmosphere that includes suspended cloud condensate in
the working fluid; the special case of a dry atmosphere is obtained for zero
specific humidities (or simply by omitting the optional specific humidity
arguments in the functions that are needed for a dry atmosphere). The
general formulation assumes that there are tracers for specific humidity
`q`, partitioned into

 - `q.tot` total water specific humidity
 - `q.liq` liquid specific humidity
 - `q.ice` ice specific humidity

to characterize the thermodynamic state and composition of moist air.

There are several types of functions:

1. Equation of state (ideal gas law):
    * `air_pressure`
2. Specific gas constant and isobaric and isochoric specific heats of moist
   air:
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
6. Functions to compute temperatures and partitioning of water into phases in
   thermodynamic equilibrium (when Gibbs' phase rule implies that the entire
   thermodynamic state of moist air, including the liquid and ice specific
   humidities, can be calculated from the 3 thermodynamic state variables, such
   as energy, pressure, and total specific humidity)
    * `liquid_fraction` (fraction of condensate that is liquid)
    * `saturation_adjustment` (compute temperature from energy, density, and
      total specific humidity)
7. Auxiliary functions for diagnostic purposes, e.g., other thermodynamic
quantities
    * `liquid_ice_pottemp` (liquid-ice potential temperature)

A moist dynamical core that assumes equilibrium thermodynamics can be
obtained from a dry dynamical core with total energy as a prognostic
variable by including a tracer for the total specific humidity `q.tot`,
using the functions, e.g., for the energies in the module, and computing
the temperature `T` and the liquid and ice specific humidities (`q.liq` and
`q.ice`) from the internal energy `e_int` by saturation adjustment.

## Usage

Users are encouraged to first establish a thermodynamic state with one of our
[Thermodynamic State Constructors](@ref). For example, we would construct
a moist thermodynamic state using

```julia
ts = PhaseEquil(param_set, e_int, ρ, q_tot);
```

here, `ρ` is the density of the moist air, and the internal energy `e_int =
e_tot - e_kin - geopotential` is the total energy `e_tot` minus kinetic energy
`e_kin` and potential energy `geopotential` (all energies per unit mass). Once
we've established a thermodynamic state, we can call [Thermodynamic state
methods](@ref) that support thermodynamic states:

```julia
T = air_temperature(ts);
q = PhasePartition(ts);
```

No changes to the "right-hand sides" of the dynamical equations are needed
for a moist dynamical core that supports clouds, as long as they do not
precipitate. Additional source-sink terms arise from precipitation.

Schematically, the workflow in such a core would look as follows:
```julia
# initialize
geopotential = grav * z
q_tot          = ...
ρ            = ...

(u, v, w)    = ...
e_kin           = 0.5 * (u^2 + v^2 + w^2)

e_tot        = total_energy(e_kin, geopotential, T, q_tot)

do timestep   # timestepping loop

  # advance dynamical variables by a timestep (temperature typically
  # appears in terms on the rhs, such as radiative transfer)
  advance(u, v, w, ρ, e_tot, q_tot)

  # compute internal energy from dynamic variables
  e_int = e_tot - 0.5 * (u^2 + v^2 + w^2) - geopotential

  # compute temperature, pressure and condensate specific humidities,
  ts = PhaseEquil(param_set, e_int, ρ, q_tot);
  T = air_temperature(ts);
  q = PhasePartition(ts);
  p = air_pressure(ts);

end
```

For a dynamical core that additionally uses the liquid and ice specific
humidities `q.liq` and `q.ice` as prognostic variables, and thus explicitly
allows the presence of non-equilibrium phases such as supercooled water,
the saturation adjustment in the above workflow is replaced calling a
non-equilibrium moist thermodynamic state:

```julia
q_tot, q_liq, q_ice = ...
ts = PhaseNonEquil(param_set, e_int, ρ, PhasePartition(q_tot, q_liq, q_ice));
T = air_temperature(ts);
p = air_pressure(ts);
```

## Extending

If MoistThermodynamics.jl does not have a particular thermodynamic
constructor that is needed, you can implement a new one in
`src/Common/MoistThermodynamics/states.jl`. In this constructor, you must
add whichever arguments you wish to offer as inputs, then translate this
thermodynamic state into one of:

 - `PhaseDry` a dry thermodynamic state, uniquely determined by two
   independent thermodynamic properties
 - `PhaseEquil` a moist thermodynamic state in thermodynamic equilibrium,
   uniquely determined by three independent thermodynamic properties
 - `PhaseNonEquil` a moist thermodynamic state in thermodynamic
   non-equilibrium, uniquely determined by four independent thermodynamic
   properties

For example, to add a thermodynamic state constructor that accepts temperature,
density and total specific humidity, we could add the following code to
`states.jl`:

```
"""
    TemperatureSHumEquil_given_density(param_set, T, ρ, q_tot)

Constructs a [`PhaseEquil`](@ref) thermodynamic state from temperature.

 - `param_set` parameter set, used to dispatch planet parameter function calls
 - `T` temperature
 - `ρ` density
 - `q_tot` total specific humidity
"""
function TemperatureSHumEquil(
    param_set::APS,
    T::FT,
    ρ::FT,
    q_tot::FT,
) where {FT <: Real}
    q = PhasePartition_equil(param_set, T, ρ, q_tot)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q_tot, T)
end
```
