# [ODESolvers](@id ODESolvers-docs)

```@meta
CurrentModule = ClimateMachine
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
```

## Multi-rate Runge Kutta Methods

```@docs
ODESolvers.MultirateRungeKutta
```

## Generic Callbacks

```@docs
GenericCallbacks
GenericCallbacks.EveryXWallTimeSeconds
GenericCallbacks.EveryXSimulationSteps
```

## ODE Solvers

```@docs
ODESolvers.solve!
ODESolvers.gettime
ODESolvers.updatedt!
```
