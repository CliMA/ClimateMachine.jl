# [ODESolvers](@id ODESolvers-docs)

```@meta
CurrentModule = ClimateMachine
```

```@docs
ODESolvers
```

## Low Storage Runge Kutta methods

```@docs
ODESolvers.LowStorageRungeKutta2N
ODESolvers.LSRK54CarpenterKennedy
ODESolvers.LSRK144NiegemannDiehlBusch
```

## Low Storage (3N) Runge Kutta methods

```@docs
ODESolvers.LowStorageRungeKutta3N
ODESolvers.LS3NRK44Classic
ODESolvers.LS3NRK33Heuns
```

## Strong Stability Preserving RungeKutta methods

```@docs
ODESolvers.StrongStabilityPreservingRungeKutta
ODESolvers.SSPRK33ShuOsher
ODESolvers.SSPRK34SpiteriRuuth
```

## Additive Runge Kutta methods

```@docs
ODESolvers.AdditiveRungeKutta
ODESolvers.ARK1ForwardBackwardEuler
ODESolvers.ARK2ImplicitExplicitMidpoint
ODESolvers.ARK2GiraldoKellyConstantinescu
ODESolvers.ARK548L2SA2KennedyCarpenter
ODESolvers.ARK437L2SA1KennedyCarpenter
ODESolvers.Trap2LockWoodWeller
ODESolvers.SSPRK22Ralstons
ODESolvers.SSPRK22Heuns
ODESolvers.LSRKEulerMethod
```

## Multi-rate Runge Kutta Methods

```@docs
ODESolvers.MultirateRungeKutta
```

## Multi-rate Infinitesimal Step Methods

```@docs
ODESolvers.TimeScaledRHS
ODESolvers.MultirateInfinitesimalStep
ODESolvers.MISRK1
ODESolvers.MIS2
ODESolvers.MISRK2a
ODESolvers.MISRK2b
ODESolvers.MIS3C
ODESolvers.MISRK3
ODESolvers.MIS4
ODESolvers.MIS4a
ODESolvers.MISKWRK43
ODESolvers.TVDMISA
ODESolvers.TVDMISB
```

## Split-explicit methods

```@docs
ODESolvers.SplitExplicitSolver
```

## GARK methods

```@docs
ODESolvers.MRIGARKESDIRK46aSandu
ODESolvers.MRIGARKIRK21aSandu
ODESolvers.MRIGARKESDIRK24LSA
ODESolvers.MRIGARKESDIRK34aSandu
ODESolvers.MRIGARKERK45aSandu
ODESolvers.MRIGARKExplicit
ODESolvers.MRIGARKESDIRK23LSA
ODESolvers.MRIGARKERK33aSandu
ODESolvers.MRIGARKDecoupledImplicit
```

## Euler methods

```@docs
ODESolvers.LinearBackwardEulerSolver
ODESolvers.AbstractBackwardEulerSolver
ODESolvers.NonLinearBackwardEulerSolver
```

## Differential Equations

```@docs
ODESolvers.DiffEqJLIMEXSolver
ODESolvers.DiffEqJLSolver
```

## ODE Solvers

```@docs
ODESolvers.solve!
ODESolvers.updatedt!
ODESolvers.gettime
ODESolvers.getsteps
```

## Generic Callbacks

```@docs
GenericCallbacks
GenericCallbacks.AtInit
GenericCallbacks.AtInitAndFini
GenericCallbacks.EveryXWallTimeSeconds
GenericCallbacks.EveryXSimulationTime
GenericCallbacks.EveryXSimulationSteps
```
