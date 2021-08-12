# Moisture model choices in Atmos.jl

The moisture model in `Atmos.jl` describes the behavior
  of suspended water in the atmosphere (i.e. water vapor,
  cloud liquid water and cloud ice).
There are three options available: `DryModel`, `EquilMoist` and `NonEquilMoist`.


## DryModel

The `DryModel` assumes that air is dry and there is no water available
  in any form.
It does not add any water related variables to state variables.


## EquilMoist

The `EquilMoist` assumes that water is present in the air and that
  all of its phases are always in thermodynamic equilibrium.
It employs an iterative search algorithm called `saturation adjustment` to find
  the temperature at which all water phases are in equilibrium
  and the corresponding `q_liq` and `q_ice`
  (cloud liquid water cloud ice specific humidities).
It adds `ρ q_tot` (air density times total water specific humidity)
   to state variables.
The `q_liq` and `q_ice` are diagnosed during output based on the temperature
  stored in the auxiliary state.


## NonEquilMoist

The `NonEquilMoist` assumes that water is present in the air,
  its phases are in thermal equilibrium (i.e., they have the same temperature),
  but it does not assume that the partitioning of water
  into its vapor, liquid, and ice phases is in equilibrium.
At each time step, the cloud liquid water and cloud ice source/sink terms
  are obtained as relaxation towards equilibrium over time scale
  that may eventually depend on factors such as the number of cloud condensation nuclei.
  (for details see [here](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/#Cloud-water-condensation/evaporation).)
This approach does not require employing iterative algorithm and
  allows for supersaturation to be present in the computational domain.
It adds `ρ q_tot`, `ρ q_liq` and `ρ q_ice` to the state variables
  (air density times total water, cloud liquid water and cloud ice specific humidities).
Because the assumed relaxation timescale for condensation/evaporation and
  deposition/sublimation is small, it requires care when choosing the
  model timestep length.


## Implementation notes

!!! warn
    While `recover_thermo_state` is an ideal long-term solution,
    right now we are directly calling new_thermo_state to avoid
    inconsistent aux states in kernels where the aux states are
    out of sync with the boundary state.

The moisture models rely on the
  [new\_thermo\_state](https://clima.github.io/ClimateMachine.jl/latest/APIs/Atmos/AtmosModel/#ClimateMachine.Atmos.new_thermo_state)
  and
  [recover\_thermo\_state](https://clima.github.io/ClimateMachine.jl/latest/APIs/Atmos/AtmosModel/#ClimateMachine.Atmos.recover_thermo_state)
  convenience functions to create a `struct` that stores
  air properties needed to uniquely define air thermodynamic state.
For the `DryModel` it is the
  [PhaseDry](https://clima.github.io/Thermodynamics.jl/dev/API/#Thermodynamics.PhaseDry)
  `struct` that has three fields:
  parameter set used by the `Atmos.jl` model, internal energy and air density.
For the `EquilMoist` model it is the
  [PhaseEquil](https://clima.github.io/Thermodynamics.jl/dev/API/#Thermodynamics.PhaseEquil)
  `struct` that has five fields:
  parameter set used by the `Atmos.jl` model, internal energy, air density,
  total water specific humidity and temperature at which all water phases
  are in equilibrium.
For the `NonEquilMoist` model it is the
  [PhaseNonEquil](https://clima.github.io/Thermodynamics.jl/dev/API/#Thermodynamics.PhaseNonEquil)
  `struct` that has four fields:
  parameter set used by the `Atmos.jl` model, internal energy,
  air density and phase partition `struct`.
All other properties, such as the speed of sound in the air,
  water vapor specific humidity,
  etc, should be computed based on the thermodynamic state `struct`.

The `new_thermo_state` function is called at the beginning of each time step
  in the `atmos_nodal_update_auxiliary_state` function.
For the `EquilMoist` model the `new_thermo_state` function calls
  the `saturation_adjustment` to find the equilibrium temperature.
It populates the fields of the `PhaseEquil` `struct`
  and saves the equilibrium air temperature into the auxiliary state.
For the `DryModel` and `NonEquilMoist` model the thermodynamic state `struct`
  is created based on the current state variables.
The `recover_thermo_state` function should be used throughout the code to create
  an instance of the thermodynamic state `struct`.
For the `EquilMoist` model it populates the `PhaseEquil` fields based on the
  current state variables and the temperature stored in the auxiliary state.
This avoids executing unnecessarily the `saturation_adjustemnt` algorithm.
For the `DryModel` and `NonEquilMoist` model the `recover_thermo_state`
  function is the same as the `new_thermo_state` function
  and populates the corresponding `struct` fields based on
  the current state variables.
