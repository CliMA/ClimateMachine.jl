module LinearSolvers

using ..MPIStateArrays

using LinearAlgebra

# just for testing LinearSolvers
LinearAlgebra.norm(A::Array, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::Array, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::Array, B::Array, weighted) = dot(A, B)

export linearsolve!, settolerance!

abstract type AbstractLinearSolver end

abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

settolerance!(solver::AbstractIterativeLinearSolver, tolerance) =
  (solver.tolerance[1] = tolerance)

doiteration!(Q, solver::AbstractIterativeLinearSolver, tolerance) =
  throw(MethodError(doiteration!, (Q, solver, tolerance))) 

initialize!(Q, Qrhs, solver::AbstractIterativeLinearSolver) =
  throw(MethodError(initialize!, (Q, Qrhs, solver))) 

"""
    linearsolve!(Q, solver::AbstractIterativeLinearSolver, tolerance)

Solves a linear problem using the `solver` with the specifed `tolerance`
storing the solution in `Q`.
"""
function linearsolve!(linearoperator!, Q, Qrhs, solver::AbstractIterativeLinearSolver)
  converged = false
  iters = 0

  initialize!(linearoperator!, Q, Qrhs, solver)

  while !converged
    converged, residual_norm = doiteration!(linearoperator!, Q, solver)
    iters += 1
  end
  
  iters
end

end
