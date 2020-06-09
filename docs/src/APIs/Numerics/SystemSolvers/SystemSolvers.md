# System Solvers

```@meta
CurrentModule = ClimateMachine.SystemSolvers
```

## Generalized Conjugate Residual Method

```@docs
GeneralizedConjugateResidual
```

## Generalized Minimal Residual Method

```@docs
GeneralizedMinimalResidual
```

## Batched Generalized Minimal Residual Method

```@docs
BatchedGeneralizedMinimalResidual
```

## Conjugate Gradient Solver Method
```@docs
ConjugateGradient
initialize!
doiteration!
```

## Shared components

```@docs
AbstractSystemSolver
AbstractIterativeSystemSolver
linearsolve!
settolerance!
prefactorize
```
