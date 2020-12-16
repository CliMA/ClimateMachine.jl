# Ocean Base Module

```@meta
CurrentModule = ClimateMachine.Ocean
```

# Hydrostatic Boussinesq

```@meta
CurrentModule = ClimateMachine.Ocean.HydrostaticBoussinesq
```

## Models

```@docs
HydrostaticBoussinesqModel
LinearHBModel
```

## BCs

```@docs
OceanBC
Penetrable
Impenetrable
NoSlip
FreeSlip
KinematicStress
Insulating
TemperatureFlux
```
# ShallowWater

```@meta
CurrentModule = ClimateMachine.Ocean.ShallowWater
```

## Models

```@docs
ShallowWaterModel
```

# OceanProblems

```@meta
CurrentModule = ClimateMachine.Ocean.OceanProblems
```

## Problems

```@docs
SimpleBox
HomogeneousBox
OceanGyre
```

## Other (development)

```@meta
CurrentModule = ClimateMachine.Ocean
```

```@docs
HydrostaticBoussinesqSuperModel
```

```@meta
CurrentModule = ClimateMachine.Ocean.JLD2Writers
```

```@docs
JLD2Writer
```

```@meta
CurrentModule = ClimateMachine.Ocean.Domains
```

```@docs
RectangularDomain
```

```@meta
CurrentModule = ClimateMachine.Ocean.Fields
```

```@docs
SpectralElementField
RectangularElement
assemble
```

```@meta
CurrentModule = ClimateMachine.Ocean.SplitExplicit01
```

```@docs
SplitExplicitLSRK2nSolver
SplitExplicitLSRK3nSolver
OceanDGModel
```
