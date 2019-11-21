"""
# RootSolvers

Module containing functions for solving roots of non-linear
equations. See [`find_zero`](@ref).


## Example

```
using CLIMA.Utilities.RootSolvers

sol = find_zero(x -> x^2 - 100^2, 0.0, 1000.0, SecantMethod())
x_root = sol.root
converged = sol.converged
```

"""
module RootSolvers

using DocStringExtensions

# disable to reduce overhead
const VERBOSE_RESULTS = false

export find_zero, SecantMethod, RegulaFalsiMethod, NewtonsMethod
export RootResults

import ForwardDiff

abstract type RootSolvingMethod end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

struct SecantMethod <: RootSolvingMethod end
struct RegulaFalsiMethod <: RootSolvingMethod end
struct NewtonsMethod <: RootSolvingMethod end

if VERBOSE_RESULTS

"""
    RootResults

Root solver results

# Fields
$(DocStringExtensions.FIELDS)
"""
struct RootResults{FT}
  "solution, ``x^*`` of ``f(x^*) = 0``"
  root::FT
  "indicates convergence"
  converged::Bool
  "the error (residual) of the roots equation (``f(x^*) = err``)"
  err::FT
  "number of iterations performed"
  iter_performed::Int
  "history of the error of the root (``f(x^*)``) per iteration"
  root_history::Vector{FT}
  "history of the root (`x`) per iteration"
  err_history::Vector{FT}
end

else

"""
    RootResults

Root solver results

# Fields
$(DocStringExtensions.FIELDS)
"""
struct RootResults{FT}
  "solution, ``x^*`` of ``f(x^*) = 0``"
  root::FT
  "indicates convergence"
  converged::Bool
end

end

# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    sol = find_zero(f, x0[, x1], method, solutiontype,
                    xatol=1e-3,
                    maxiters=10_000)

Finds the nearest root of `f` to `x0` and `x1`. Returns a the value of the root `x` such
that `f(x) ≈ 0`, and a Boolean value `converged` indicating convergence.

`method` can be one of:
- `SecantMethod()`: [Secant method](https://en.wikipedia.org/wiki/Secant_method)
- `RegulaFalsiMethod()`: [Regula Falsi method](https://en.wikipedia.org/wiki/False_position_method#The_regula_falsi_(false_position)_method).
- `NewtonsMethod()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
  - The `x1` argument is omitted for Newton's method.

The keyword arguments:
- `xatol` is the absolute tolerance of the input.
- `maxiters` is the maximum number of iterations.
"""
function find_zero end

function find_zero(f::F, x0::FT, x1::FT, ::SecantMethod,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  if VERBOSE_RESULTS
    x_history = FT[x0]
    y_history = FT[y0]
  end
  for i in 1:maxiters
    Δx = x1 - x0
    Δy = y1 - y0
    x0, y0 = x1, y1
    x1 -= y1 * Δx / Δy
    y1 = f(x1)
    if VERBOSE_RESULTS
      push!(x_history, x1)
      push!(y_history, y1)
    end
    if abs(x0-x1) <= xatol
      return VERBOSE_RESULTS ? RootResults(x1, true, y1, i, x_history, y_history) :
                               RootResults(x1, true)
    end
  end
  return VERBOSE_RESULTS ? RootResults(x1, false, y1, maxiters, x_history, y_history) :
                           RootResults(x1, false)
end

function find_zero(f::F, x0::FT, x1::FT, ::RegulaFalsiMethod,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  if VERBOSE_RESULTS
    x_history = FT[x0]
    y_history = FT[y0]
  end
  @assert y0 * y1 < 0
  lastside = 0
  local x, y
  for i in 1:maxiters
    x = (x0 * y1 - x1 * y0)/ (y1 - y0)
    y = f(x)
    if VERBOSE_RESULTS
      push!(x_history, x)
      push!(y_history, y)
    end
    if y * y0 < 0
      if abs(x-x1) <= xatol
        return VERBOSE_RESULTS ? RootResults(x, true, y, i, x_history, y_history) :
                                 RootResults(x, true)
      end
      x1, y1 = x, y
      if lastside == +1
        y0 /= 2
      end
      lastside = +1
    else
      if abs(x0-x) <= xatol
        return VERBOSE_RESULTS ? RootResults(x, true, y, i, x_history, y_history) :
                                 RootResults(x, true)
      end
      x0, y0 = x, y
      if lastside == -1
        y1 /= 2
      end
      lastside = -1
    end
  end
  return VERBOSE_RESULTS ? RootResults(x, false, y, maxiters, x_history, y_history) :
                           RootResults(x, false)
end


"""
    value_deriv(f, x)

Compute the value and derivative `f(x)` using ForwardDiff.jl.
"""
function value_deriv(f, x::FT) where {FT}
    tag = typeof(ForwardDiff.Tag(f, FT))
    y = f(ForwardDiff.Dual{tag}(x,one(x)))
    ForwardDiff.value(tag, y), ForwardDiff.partials(tag, y, 1)
end

function find_zero(f::F, x0::FT, ::NewtonsMethod,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  local y
  if VERBOSE_RESULTS
    x_history = FT[x0]
    y,y′ = value_deriv(f, x0)
    y_history = FT[y]
  end
  for i in 1:maxiters
    y,y′ = value_deriv(f, x0)
    x1 = x0 - y/y′
    if VERBOSE_RESULTS
      push!(x_history, x1)
      push!(y_history, y)
    end
    if abs(x0-x1) <= xatol
      return VERBOSE_RESULTS ? RootResults(x1, true, y, i, x_history, y_history) :
                               RootResults(x1, true)
    end
    x0 = x1
  end
  return VERBOSE_RESULTS ? RootResults(x0, false, y, maxiters, x_history, y_history) :
                           RootResults(x0, false)
end

end
