"""
RootSolvers.jl provides methods for solving roots of non-linear equations.
"""
module RootSolvers

import Base

export find_zero
export IterParams

export SecantMethod

struct IterParams{R<:AbstractFloat, I<:Int}
  tol_abs::R
  iter_max::I
end

abstract type RootSolvingMethod end
struct SecantMethod<:RootSolvingMethod end

"""
find_zero(f, x_0, x_1, args, iter_params, method)

find_zero returns the root of function `f(x, args...)`, given
`initial_guess`, and `iter_params`. The method used to solve for the root
is provided by `method`. The returned result is a tuple of the root and
a Bool indicating convergence.

To braodcast this function, wrap args that need not be broadcasted in the
function Ref().

  using RootSolvers
  x_0 = 0.0
  x_1 = 1.0
  f(x, y) = x^2 - y
  x_star2 = 10000.0
  args = Tuple(x_star2)
  x_star = sqrt(x_star2)
  tol_abs = 1.0e-3
  iter_max = 100

  x_root, converged = find_zero(f,
                                x_0,
                                x_1,
                                args,
                                IterParams(tol_abs, iter_max),
                                SecantMethod())

  # Broadcasted example (when x_0 and x_1 are arrays):

  x_root, converged = find_zero.(f,
                                 x_0,
                                 x_1,
                                 Ref(args),
                                 Ref(IterParams(tol_abs, iter_max)),
                                 Ref(SecantMethod()))
"""
function find_zero(f::F,
                   x_0::T,
                   x_1::T,
                   args::Tuple,
                   iter_params::IterParams{R, Int},
                   method::SecantMethod
                   )::Tuple{T, Bool} where {F, R, T <: Union{R, AbstractArray{R}}}
  iter_max = iter_params.iter_max
  tol_abs = iter_params.tol_abs
  Δx = x_1 - x_0
  y_0 = f(x_0, args...)
  y_1 = f(x_1, args...)
  x = x_1 - y_1 * Δx / (y_1 - y_0) # puts x in outer scope, & saves iter_max = 0 case
  for i in 1:iter_max
    x = x_1 - y_1 * Δx / (y_1 - y_0)
    y_0 = y_1
    x_0 = x_1
    x_1 = x
    y_1 = f(x_1, args...)
    Δx = x_1 - x_0
    if abs(y_1) < tol_abs
      return x, true
    end
  end
  return x, false
end

end