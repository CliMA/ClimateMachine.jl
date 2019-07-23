module LinearSolvers

using ..MPIStateArrays

using LinearAlgebra

# just for testing LinearSolvers
LinearAlgebra.norm(A::Array, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::Array, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::Array, B::Array, weighted) = dot(A, B)

export linearsolve!, settolerance!
export AbstractLinearSolver, AbstractIterativeLinearSolver

abstract type AbstractLinearSolver end

abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

"""
    settolerance!(solver::AbstractIterativeLinearSolver, tolerance)

Sets the tolerance of the iterative linear solver `solver` to `tolerance`.
"""
settolerance!(solver::AbstractIterativeLinearSolver, tolerance) =
  (solver.tolerance[1] = tolerance)

doiteration!(Q, solver::AbstractIterativeLinearSolver, tolerance) =
  throw(MethodError(doiteration!, (Q, solver, tolerance))) 

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

  initialize!(linearoperator!, Q, Qrhs, solver)

  while !converged
    converged, residual_norm = doiteration!(linearoperator!, Q, solver)
    iters += 1
    if !isfinite(residual_norm)
      error("norm of residual is not finite after $iters iterations of `doiteration!`")
    end
  end
  
  iters
end

end
