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
ODESolvers.SSPRK22Ralstons
ODESolvers.SSPRK22Heuns
ODESolvers.LSRKEulerMethod
```

## Multi-rate Runge Kutta Methods

```@docs
ODESolvers.MultirateRungeKutta
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
```

## ODE Solvers

```@docs
ODESolvers.solve!
ODESolvers.gettime
ODESolvers.updatedt!
```

## Generic Callbacks

```@docs
GenericCallbacks
GenericCallbacks.AtStart
GenericCallbacks.EveryXWallTimeSeconds
GenericCallbacks.EveryXSimulationTime
GenericCallbacks.EveryXSimulationSteps
```
