module RootSolvers

export find_zero
export IterParams

export InitialGuessSecant
export SecantMethod

struct IterParams{R<:Real, I<:Int}
  tol_abs::R
  iter_max::I
end

abstract type RootSolvingMethod end
struct SecantMethod<:RootSolvingMethod end

abstract type InitialGuess end
struct InitialGuessSecant{R<:Real} <: InitialGuess
  x_0::R
  x_1::R
end

"""
find_zero(f, initial_guess, args, iter_params, method)

find_zero returns the root of function `f` that takes arguments `args`,
`initial_guess`, and `iter_params`. The method used to solve for the root
is provided by `method`. The returned result is a tuple of the root and
a Bool indicating convergence.

"""

function find_zero(f::Function,
                   initial_guess::InitialGuessSecant{R},
                   args::Tuple{Vararg{R}},
                   iter_params::IterParams{R, I},
                   method::SecantMethod
                   )::Tuple{R, Bool} where {R<:Real, I<:Int}
  x_1 = initial_guess.x_1
  x_0 = initial_guess.x_0
  iter_max = iter_params.iter_max
  tol_abs = iter_params.tol_abs

  Δx = x_1 - x_0
  y_0 = f(x_0, args...)
  y_1 = f(x_1, args...)
  x = x_1 - y_1 * Δx / (y_1 - y_0)
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