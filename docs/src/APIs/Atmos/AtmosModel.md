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

## AtmosModel methods

```@docs
ClimateMachine.BalanceLaws.flux_first_order!(m::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real, direction)
ClimateMachine.BalanceLaws.flux_second_order!(atmos::AtmosModel, flux::Grad, state::Vars, diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
ClimateMachine.BalanceLaws.init_state_auxiliary!(m::AtmosModel, state_auxiliary::MPIStateArray, grid, direction)
ClimateMachine.BalanceLaws.source!(m::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real, direction)
ClimateMachine.BalanceLaws.init_state_prognostic!(m::AtmosModel, state::Vars, aux::Vars, localgeo, t, args...)
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
