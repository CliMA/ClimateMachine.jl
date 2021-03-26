# [Using Diagnostics](@id How-to-diagnostics)

```@meta
CurrentModule = ClimateMachine
```

An experiment can configure ClimateMachine to output various diagnostic
variables to NetCDF files at specifiable intervals during a simulation.
To do so, it must create a
[`ClimateMachine.DiagnosticsConfiguration`](@ref) which is passed to
[`ClimateMachine.invoke!`](@ref) with the `diagnostics_config` keyword.

```@meta
CurrentModule = ClimateMachine.Diagnostics
```

A `DiagnosticsConfiguration` is constructed with a list of the
[`DiagnosticsGroup`](@ref)s of interest. The `DiagnosticsGroup`s
currently defined together with the functions used to construct them
are:

- `AtmosGCMDefault` and `AtmosLESDefault` -- [`setup_atmos_default_diagnostics`](@ref)
- `AtmosLESCore` -- [`setup_atmos_core_diagnostics`](@ref)
- `AtmosLESDefaultPerturbations` -- [`setup_atmos_default_perturbations`](@ref)
- `AtmosRefStatePerturbations` -- [`setup_atmos_refstate_perturbations`](@ref)
- `AtmosTurbulenceStats` -- [`setup_atmos_turbulence_stats`](@ref)
- `AtmosMassEnergyLoss` -- [`setup_atmos_mass_energy_loss`](@ref)
- `AtmosLESSpectra` and `AtmosGCMSpectra` -- [`setup_atmos_spectra_diagnostics`](@ref)
- `DumpState` -- [`setup_dump_state_diagnostics`](@ref)
- `DumpAux` -- [`setup_dump_aux_diagnostics`](@ref)
- `DumpTendencies` -- [`setup_dump_tendencies_diagnostics`](@ref)

Each of these diagnostics groups contains a set of diagnostic
variables.

```@meta
CurrentModule = ClimateMachine
```

Users can define their own diagnostics groups, and the
[`ClimateMachine.DiagnosticsMachine`](@ref), currently in development,
provides functionality to simplify doing so.

```@meta
CurrentModule = ClimateMachine.Diagnostics
```

Here is a code snippet adapted from the Taylor-Green vortex experiment,
showing the creation of three diagnostics groups and their use:
```julia
...
using ClimateMachine.Diagnostics
...
    ts_dgngrp = setup_atmos_turbulence_stats(
        AtmosLESConfigType(),
        "360steps",
        driver_config.name,
        tnor,
        titer,
    )

    boundaries = [
        xmin ymin zmin
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    ds_dgngrp = setup_atmos_spectra_diagnostics(
        AtmosLESConfigType(),
        "0.06ssecs",
        driver_config.name,
        interpol = interpol,
        snor,
    )

    me_dgngrp = setup_atmos_mass_energy_loss(
        AtmosLESConfigType(),
        "0.02ssecs",
        driver_config.name,
    )

    dgn_config = ClimateMachine.DiagnosticsConfiguration([
        ts_dgngrp,
        ds_dgngrp,
        me_dgngrp,
    ],)
...
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        check_euclidean_distance = true,
    )
```

When this experiment is run with the command line argument
`--diagnostics=default`, three NetCDF files are created, one for each
group. The `AtmosLESSpectra` diagnostic variables are output on an
interpolated grid, as specified above. Each group has a different
interval specified, thus the number of entries in each NetCDF file
(along the unlimited `time` dimension) will differ.

When designing customized diagnostics groups, please use the above
example as a template and refer to the [list of current diagnostics
variables](@ref Diagnostics-vars).
