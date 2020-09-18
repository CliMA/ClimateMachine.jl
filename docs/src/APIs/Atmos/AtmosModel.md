# [AtmosModel](@id AtmosModel-docs)

```@meta
CurrentModule = ClimateMachine
```

## AtmosProblem

```@docs
ClimateMachine.Atmos.AtmosProblem
```

## AtmosModel balance law

```@docs
ClimateMachine.Atmos.AtmosModel
```

## Reference states

```@docs
ClimateMachine.Atmos.HydrostaticState
ClimateMachine.Atmos.InitStateBC
ClimateMachine.Atmos.ReferenceState
ClimateMachine.Atmos.NoReferenceState
```

## Thermodynamics

```@docs
ClimateMachine.Atmos.recover_thermo_state
ClimateMachine.Atmos.new_thermo_state
```

## Moisture

```@docs
ClimateMachine.Atmos.DryModel
ClimateMachine.Atmos.EquilMoist
ClimateMachine.Atmos.NonEquilMoist
ClimateMachine.Atmos.NoPrecipitation
ClimateMachine.Atmos.Rain
```

## Stabilization

```@docs
ClimateMachine.Atmos.RayleighSponge
```

## BCs

```@docs
ClimateMachine.Atmos.AtmosBC
ClimateMachine.Atmos.DragLaw
ClimateMachine.Atmos.Impermeable
ClimateMachine.Atmos.PrescribedMoistureFlux
ClimateMachine.Atmos.BulkFormulaMoisture
ClimateMachine.Atmos.FreeSlip
ClimateMachine.Atmos.PrescribedTemperature
ClimateMachine.Atmos.PrescribedEnergyFlux
ClimateMachine.Atmos.BulkFormulaEnergy
ClimateMachine.Atmos.ImpermeableTracer
ClimateMachine.Atmos.Impenetrable
ClimateMachine.Atmos.Insulating
ClimateMachine.Atmos.NoSlip
ClimateMachine.Atmos.average_density
```

## Sources

```@docs
ClimateMachine.Atmos.RemovePrecipitation
ClimateMachine.Atmos.CreateClouds
```
