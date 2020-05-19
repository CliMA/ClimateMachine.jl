# Atmosphere model configurations

The struct `AtmosModel` defines a specific subtype of a balance law
(i.e. conservation equations) specific to atmospheric modelling. A
complete description of a `model` is provided by the fields listed
below. In this implementation of the `AtmosModel` we concern ourselves
with the conservative form of the compressible equations of moist fluid
motion given a set of initial, boundary and forcing(source) conditions.

### [LES Configuration](@id LESConfig) (with defaults)
Default field values for the LES `AtmosModel` definition are included
below. Users are directed to the model subcomponent pages to view the
possible options for each subcomponent.
```
    ::Type{AtmosLESConfigType},
    param_set::AbstractParameterSet;
    orientation::O = FlatOrientation(),
    ref_state::RS = HydrostaticState(
        LinearTemperatureProfile(
            FT(200),
            FT(280),
            FT(grav(param_set)) / FT(cp_d(param_set)),
        ),
        FT(0),
    ),
    turbulence::T = SmagorinskyLilly{FT}(0.21),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis(), GeostrophicForcing{FT}(7.62e-5, 0, 0)),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
```

!!! note

    Most AtmosModel subcomponents are common to both LES / GCM
    configurations.  Equation sets are written in vector-invariant form and
    solved in Cartesian coordinates.  The component `orientation` determines
    whether the problem is solved in a `box (LES)` or a `sphere (GCM)`)


### [GCM Configuration](@id GCMConfig)(with defaults)
Default field values for the GCM `AtmosModel` definition are included
below. Users are directed to the model subcomponent pages to view the
possible options for each subcomponent.

```
    ::Type{AtmosGCMConfigType},
    param_set::AbstractParameterSet;
    orientation::O = SphericalOrientation(),
    ref_state::RS = HydrostaticState(
        LinearTemperatureProfile(
            FT(200),
            FT(280),
            FT(grav(param_set)) / FT(cp_d(param_set)),
        ),
        FT(0),
    ),
    turbulence::T = SmagorinskyLilly{FT}(C_smag(param_set)),
    hyperdiffusion::HD = NoHyperDiffusion(),
    moisture::M = EquilMoist{FT}(),
    precipitation::P = NoPrecipitation(),
    radiation::R = NoRadiation(),
    source::S = (Gravity(), Coriolis()),
    tracers::TR = NoTracers(),
    boundarycondition::BC = AtmosBC(),
    init_state_conservative::IS = nothing,
    data_config::DC = nothing,
```
