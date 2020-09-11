# System Solvers

```@meta
CurrentModule = ClimateMachine.SystemSolvers
```

## Non-linear solvers

```@docs
LSOnly
JacobianAction
JacobianFreeNewtonKrylovSolver
```

## Linear solvers

### Generalized Conjugate Residual Method

```@docs
GeneralizedConjugateResidual
```

### Generalized Minimal Residual Method

```@docs
GeneralizedMinimalResidual
```

### Batched Generalized Minimal Residual Method

```@docs
BatchedGeneralizedMinimalResidual
```

### Conjugate Gradient Solver Method
```@docs
ConjugateGradient
initialize!
doiteration!
```

### LU Decomposition

```@docs
ManyColumnLU
SingleColumnLU
```

## Preconditioners

```@docs
NoPreconditioner
ColumnwiseLUPreconditioner
```

## Shared components

```@docs
AbstractSystemSolver
AbstractNonlinearSolver
AbstractIterativeSystemSolver
AbstractPreconditioner
nonlinearsolve!
linearsolve!
settolerance!
prefactorize
preconditioner_update!
preconditioner_solve!
preconditioner_counter_update!
```
