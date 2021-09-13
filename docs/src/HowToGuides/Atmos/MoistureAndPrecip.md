## Representing moisture and precipitation

This tutorial shows how to setup an Atmos experiment with moisture
(i.e. water vapor, cloud liquid water, and cloud ice)
and precipitation (i.e. rain and/or snow).
It expands on the LES and GCM tutorials already available in the
[Atmos tutorials section](https://clima.github.io/ClimateMachine.jl/latest/generated/TutorialList/#Atmos).
Therefore, this tutorial only discusses additional steps that are needed
in order to define an experiment with moisture and precipitation.
For a tutorial on how to setup basic Atmos experiment see the
[Held-Suarez](https://clima.github.io/ClimateMachine.jl/latest/generated/Atmos/heldsuarez/)
or the [Rising thermal bubble](https://clima.github.io/ClimateMachine.jl/latest/generated/Atmos/risingbubble/)
tutorials.
For a full experiment setup with moisture and precipitation
see the [DyCOMS](https://github.com/CliMA/ClimateMachine.jl/blob/master/experiments/AtmosLES/dycoms.jl)
or the [Squall line](https://github.com/CliMA/ClimateMachine.jl/blob/master/experiments/AtmosLES/squall_line.jl)
setups that are part of the Atmos model CI.

When setting up a moist and precipitating Atmos experiment,
the user has to decide between two moisture and three precipitation models.
The differences between the two moisture models are described in the
[Moisture model](https://clima.github.io/ClimateMachine.jl/latest/HowToGuides/Atmos/MoistureModelChoices/) section.
In general, the `EquilMoist` model solves for only one additional state variable
and diagnoses the cloud condensate partitioning between liquid and ice
based on equilibrium assumptions and an iterative search algorithm.
The `NonEquilMoist` model solves equations for all three cloud condensate state variables
Because the condensation and deposition timescales are fast,
in many cases, the `EquilMoist` model is a good approximation.
However, it is obviously not sufficient when modeling nonequilibrium
processes, such as supercooled clouds.
Because of the lack of need for iterative search to perform partitioning
of cloud condensate, the `NonEquilMoist` model might be more robust,
if a little slower than the `EquilMoist` model.
The differences between the three precipitation models are described in the
[Precipitation model](https://clima.github.io/ClimateMachine.jl/latest/HowToGuides/Atmos/PrecipitationModelChoices/) section.
In general, the `NoPrecipitation` option can be used in simulations without
precipitation or in simulations when the effect of precipitation
is modeled by instantly removing cloud condensate.
This option, coupled with either of the moisture models, is a good start
when running in moist `AtmosGCM` configuration.
The parameterization of instantaneous precipitation removal is described by the
[0-moment microphysics](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_0M/)
scheme.
The `RainModel` encapsulates the "warm rain" processes, i.e.
the formation of precipitation in temperatures above freezing.
The `RainSnowModel` describes the full set of microphysical processes
leading to the formation of precipitation.
Both models are based on
[1-moment microphysics](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_1M/)
scheme, which is a simple representation of cloud microphysics.
It is a good starting point for the moist `AtmosLES` configurations,
coupling to the subgrid scale parameterizations, and testing the machine learning pipelines.
In the future, a more sophisticated 2-moment microphysics scheme will most likely be needed.


## Using CLIMAParameters.jl

The microphysics parameterizations used to compute the sources and sinks for
moisture and precipitation variables depends on
[CLIMAParameters.jl](https://github.com/CliMA/CLIMAParameters.jl).
A set with all the parameter values needs to be created and passed in as one
of the arguments.

```julia
using CLIMAParameters
using CLIMAParameters.Planet
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
```

After this is done, the created `param_set` should be passed
in the same way as in the "dry" tutorials.


## Initial condition

As discussed in the documentation on the
[moisture](https://clima.github.io/ClimateMachine.jl/latest/HowToGuides/Atmos/MoistureModelChoices/)
and
[precipitation](https://clima.github.io/ClimateMachine.jl/latest/HowToGuides/Atmos/PrecipitationModelChoices/)
choices, different model configurations result in different state variables.
When using the `EquilMoist` model, the initial condition on total water
is the only thing that is needed:

```julia
state.moisture.ρq_tot = ρ * 0.0042
```

When using the `NonEquilMoist` model, the initial condition on total water,
cloud liquid water and cloud ice is needed:

```julia
state.moisture.ρq_tot = ρ * 0.0042
state.moisture.ρq_liq = FT(0)
state.moisture.ρq_ice = FT(0)
```

Similarly for the `RainModel` we need to define the initial condition for
rain:

```julia
state.precipitation.ρq_rai = FT(0)
```

And for the `RainSnowModel` the initial condition for both rain and snow is needed:

```julia
state.precipitation.ρq_rai = FT(0)
state.precipitation.ρq_sno = FT(0)
```

As in the previous tutorials `FT()` is the float type chosen for the simulation.

The moisture and precipitation state variables are grouped together
into `state.moisture` and `state.precipitation` fields.
As shown when specifying the initial condition,
this hierarchy has to be observed when accessing the members
of moisture or precipitation state variables.


## Choosing the moisture and precipitation models and sources

When opting to use the `EquilMoist` moisture model, we need to specify
the maximum number of iterations and the tolerance allowed
in the iterative search performed to diagnose cloud condensate
phase partitioning.
The cloud liquid water and cloud ice are stored in the auxiliary
state variables.
Because the partitioning is diagnosed, no additional source terms
have to be specified:

```julia
moisture = EquilMoist(; maxiter = 8, tolerance = FT(1e-1))
```

Alternatively, when using the `NonEquilMoist` model, an additional source term
`CreateClouds` is needed.
It is assumed here that `source` lists all the previously
defined source terms.
We are splatting to it the additional
cloud condensate sources for cloud liquid water and cloud ice.

```julia
moisture = NonEquilMoist()
source = (source..., CreateClouds())
```

If choosing the `NoPrecipitation` model we can either define no additional
source terms (a true simulation without any representation of precipitation),
or use the instant precipitation removal source term `RemovePrecipitation`.
The boolean flag passed to it as argument chooses between two definitions
of the threshold above which cloud condensate is removed,
see [here](https://clima.github.io/CloudMicrophysics.jl/dev/Microphysics_0M/#Moisture-sink-due-to-precipitation).
The flag set to `true` results in cloud condensate based threshold.
The flag set to `false` results in saturation excess threshold.

```julia
precipitation = NoPrecipitation()
source = (source..., RemovePrecipitation(true))
```

If using the `RainModel`, the `WarmRain_1M` source terms have to be chosen:

```julia
precipitation = RainModel()
source = (source..., WarmRain_1M())
```

Alternatively, if using the `RainSnowModel`, the `RainSnow_1M` source terms
have to be chosen:

```julia
precipitation = RainSnowModel()
source = (source..., RainSnow_1M())
```

As in the previous tutorials, all of the `source`, `moisture`, `precipitation`
choices, along with the `param_set` `struct` have to be passed to
the Atmos model configuration:

```julia
physics = AtmosPhysics{FT}(
    param_set;
    ref_state = ref_state,
    moisture = moisture,
    precipitation = precipitation,
    turbulence = SmagorinskyLilly{FT}(C_smag),
)
model = AtmosModel{FT}(
    AtmosLESConfigType,
    physics;
    problem = problem,
    source = source,
)
```


## Boundary conditions

Boundary condition specification is currently undergoing a re-write.
More details on boundary conditions will follow soon.
By default no additional moisture or precipitation fluxes are applied.


## Using filters

All moisture and precipitation variables are positive definite.
However, due to numerical errors, they can turn negative during the simulation.
To mitigate that behavior it is advisable to use the `TMAR filter`.
The filter adjust the nodal points inside the DG element.
It truncates the negative nodal points to zero
and adjusts the values of the remaining points
to preserve the element average.
To conserve mass of the advected tracers the `TMAR filter` should be combined
with a flux-correction to those DG elements whose average value is negative.
The flux-corrected option is currently being implemented.
If mass conservation is deemed more important than positive sign of advected tracers,
one can run the simulation without the `TMAR filter`.
The microphysics functions are implemented such that `f(negative_argument) = f(0)`.

The list of state variables to be filtered
depends on the moisture and precipitation model choices.
For example:

```julia
filter_vars = ("moisture.ρq_tot",)
filter_vars = (filter_vars..., "moisture.ρq_liq", "moisture.ρq_ice")
filter_vars = (filter_vars..., "precipitation.ρq_rai")
filter_vars = (filter_vars..., "precipitation.ρq_rai", "precipitation.ρq_sno")
```

The list of variables to be filtered should then be passed to the `TMAR filter` `callback`:

```julia
cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
    Filters.apply!(
        solver_config.Q,
        filter_vars,
        solver_config.dg.grid,
        TMARFilter(),
    )
    nothing
end
```

And the filter `callback` should be passed to the final `invoke` method:

```julia
result = ClimateMachine.invoke!(
    solver_config;
    diagnostics_config = dgn_config,
    user_callbacks = (cbtmarfilter,),
    check_euclidean_distance = true,
)
```

## Accessing moisture and precipitation variables

To access moisture and precipitation variables at the end of the simulation
(for example for some debugging or assert checking in the CI)
we need to use the `varsindex` and `vars_state` functions:

```julia
m = driver_config.bl
Q = solver_config.Q
ρ_ind = varsindex(vars_state(m, Prognostic(), FT), :ρ)

ρq_tot_ind = varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_tot)
ρq_sno_ind = varsindex(vars_state(m, Prognostic(), FT), :precipitation, :ρq_sno)

min_q_tot = minimum(abs.(Array(Q[:, ρq_tot_ind, :]) ./ Array(Q[:, ρ_ind, :]),))
max_q_sno = maximum(abs.(Array(Q[:, ρq_sno_ind, :]) ./ Array(Q[:, ρ_ind, :]),))

@info(min_q_tot, max_q_sno)
```

The vertical profiles (horizontal averages) of moisture and precipitation
variables, their variances and covariances,
as well as the liquid, ice rain and snow water paths
are available in the output netcdf file when using the
`setup_atmos_default_diagnostics` group.
Additionally one can choose to save into the netcdf output all of the state
variables and auxiliary variables using the `setup_dump_state_diagnostics`
and `setup_dump_aux_diagnostics` groups.
