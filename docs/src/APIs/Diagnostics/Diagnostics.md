# [Diagnostics](@id Diagnostics-docs)

```@meta
CurrentModule = ClimateMachine.Diagnostics
```

```@docs
Diagnostics
```

### Types

```@docs
Diagnostics.DiagnosticsGroup
```

### [Diagnostics groups](@id Diagnostics-groups)

A `ClimateMachine` driver may use any number of the methods described below
to create `DiagnosticsGroup`s which must be specified to the `ClimateMachine`
in a `DiagnosticsConfiguration`.

```@docs
Diagnostics.setup_atmos_default_diagnostics
Diagnostics.setup_atmos_core_diagnostics
Diagnostics.setup_atmos_default_perturbations
Diagnostics.setup_atmos_refstate_perturbations
Diagnostics.setup_dump_state_diagnostics
Diagnostics.setup_dump_aux_diagnostics
```
