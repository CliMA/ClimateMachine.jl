module LinearSolvers

using ..MPIStateArrays

using StaticArrays, LinearAlgebra

# just for testing LinearSolvers
LinearAlgebra.norm(A::MVector, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::MVector, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::MVector, B::MVector, weighted) = dot(A, B)

export linearsolve!, settolerance!
export AbstractLinearSolver, AbstractIterativeLinearSolver

"""
This is an abstract type representing a generic linear solver.

"""
abstract type AbstractLinearSolver end

"""
This is an abstract type representing a generic iterative
linear solver.

The available concrete implementations are:

  - [GeneralizedConjugateResidual](@ref)
  - [GeneralizedMinimalResidual](@ref)
"""
abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    settolerance!(solver::AbstractIterativeLinearSolver, tolerance)

Sets the tolerance of the iterative linear solver `solver` to `tolerance`.
"""
settolerance!(solver::AbstractIterativeLinearSolver, tolerance) =
  (solver.tolerance[1] = tolerance)

doiteration!(Q, Qrhs, solver::AbstractIterativeLinearSolver, tolerance) =
  throw(MethodError(doiteration!, (Q, Qrhs, solver, tolerance)))

initialize!(Q, Qrhs, solver::AbstractIterativeLinearSolver) =
  throw(MethodError(initialize!, (Q, Qrhs, solver))) 

"""
    linearsolve!(linearoperator!, Q, Qrhs, solver::AbstractIterativeLinearSolver)

Solves a linear problem defined by the `linearoperator!` function and the state
`Qrhs`, i.e,

```math
  L(Q) = Q_{rhs}
```

using the `solver` and the initial guess `Q`. After the call `Q` contains the solution.
"""
function linearsolve!(linearoperator!, Q, Qrhs, solver::AbstractIterativeLinearSolver)
  converged = false
  iters = 0

  threshold = initialize!(linearoperator!, Q, Qrhs, solver)

  while !converged
    converged, inner_iters, residual_norm = 
      doiteration!(linearoperator!, Q, Qrhs, solver, threshold)

    iters += inner_iters

    if !isfinite(residual_norm)
      error("norm of residual is not finite after $iters iterations of `doiteration!`")
    end
    
    achieved_tolerance = residual_norm / threshold * solver.tolerance[1]
  end
  
  iters
end

end
