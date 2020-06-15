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

## LU Decomposition

```@docs
ManyColumnLU
SingleColumnLU
```

## Shared components

```@docs
AbstractSystemSolver
AbstractIterativeSystemSolver
linearsolve!
settolerance!
prefactorize
```
