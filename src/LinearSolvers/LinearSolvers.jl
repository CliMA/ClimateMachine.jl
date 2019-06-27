module LinearSolvers

using ..MPIStateArrays

using LinearAlgebra

# just for testing LinearSolvers
LinearAlgebra.norm(A::Array, p::Real, weighted::Bool) = norm(A, p)
LinearAlgebra.norm(A::Array, weighted::Bool) = norm(A, 2, weighted)
LinearAlgebra.dot(A::Array, B::Array, weighted) = dot(A, B)

export linearsolve!

abstract type AbstractLinearSolver end

abstract type AbstractIterativeLinearSolver <: AbstractLinearSolver end

doiteration!(Q, solver::AbstractIterativeLinearSolver, tolerance) =
  throw(MethodError(doiteration!, (Q, solver, tolerance))) 

"""
    linearsolve!(Q, solver::AbstractIterativeLinearSolver, tolerance)

Solves a linear problem using the `solver` with the specifed `tolerance`
storing the solution in `Q`.
"""
function linearsolve!(Q, solver::AbstractIterativeLinearSolver, tolerance)
  converged = false
  iters = 0

  while !converged
    converged, residual_norm = doiteration!(Q, solver, tolerance)
  end
  
  iters
end

end
