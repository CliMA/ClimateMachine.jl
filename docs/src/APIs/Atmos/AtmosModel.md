# AtmosModel

```@meta
CurrentModule = ClimateMachine
```

## AtmosModel type

```@docs
ClimateMachine.Atmos.NoOrientation
ClimateMachine.Atmos.FlatOrientation
ClimateMachine.Atmos.SphericalOrientation
ClimateMachine.Atmos.AtmosModel
```

## Turbulence / SGS
```@docs
ClimateMachine.Atmos.turbulence_tensors
ClimateMachine.Atmos.principal_invariants
ClimateMachine.Atmos.symmetrize
ClimateMachine.Atmos.norm2
ClimateMachine.Atmos.strain_rate_magnitude
ClimateMachine.Atmos.ConstantViscosityWithDivergence
ClimateMachine.Atmos.SmagorinskyLilly
ClimateMachine.Atmos.Vreman
ClimateMachine.Atmos.AnisoMinDiss
```

## Moisture

```@docs
ClimateMachine.Atmos.NoPrecipitation
ClimateMachine.Atmos.DryModel
ClimateMachine.Atmos.EquilMoist
ClimateMachine.Atmos.Rain
```

## Reference states

```@docs
ClimateMachine.Atmos.RemainderModel
ClimateMachine.Atmos.HydrostaticState
ClimateMachine.Atmos.InitStateBC
ClimateMachine.Atmos.NoReferenceState
```

## Stabilization

```@docs
ClimateMachine.Atmos.RayleighSponge
ClimateMachine.Atmos.StandardHyperDiffusion
```

## BCs
```@docs
ClimateMachine.Atmos.AtmosBC
ClimateMachine.Atmos.DragLaw
ClimateMachine.Atmos.Impermeable
ClimateMachine.Atmos.PrescribedMoistureFlux
ClimateMachine.Atmos.FreeSlip
ClimateMachine.Atmos.NoHyperDiffusion
ClimateMachine.Atmos.PrescribedTemperature
ClimateMachine.Atmos.PrescribedEnergyFlux
ClimateMachine.Atmos.ImpermeableTracer
ClimateMachine.Atmos.Impenetrable
ClimateMachine.Atmos.Insulating
ClimateMachine.Atmos.NoSlip
```
