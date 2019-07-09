"""
# RootSolvers

Module containing functions for solving roots of non-linear
equations. See [`find_zero`](@ref).


## Example

```
using CLIMA.Utilities.RootSolvers

x_root, converged = find_zero(x -> x^2 - 100^2, 0.0, 1000.0, SecantMethod())
```

"""
module RootSolvers

export find_zero, SecantMethod, RegulaFalsiMethod, NewtonsMethod

import ForwardDiff

abstract type RootSolvingMethod end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

struct SecantMethod <: RootSolvingMethod end
struct RegulaFalsiMethod <: RootSolvingMethod end
struct NewtonsMethod <: RootSolvingMethod end

# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    x, converged = find_zero(f, x0[, x1], method,
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

function find_zero(f::F, x0::T, x1::T, ::SecantMethod,
                   xatol=T(1e-3),
                   maxiters=10_000) where {F, T<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  for i in 1:maxiters
    Δx = x1 - x0
    Δy = y1 - y0
    x0, y0 = x1, y1
    x1 -= y1 * Δx / Δy
    y1 = f(x1)
    if abs(x0-x1) <= xatol
      return x1, true
    end
  end
  return x1, false
end

function find_zero(f::F, x0::T, x1::T, ::RegulaFalsiMethod,
                   xatol=T(1e-3),
                   maxiters=10_000) where {F, T<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  @assert y0 * y1 < 0
  lastside = 0
  local x
  for i in 1:maxiters
    x = (x0 * y1 - x1 * y0)/ (y1 - y0)
    y = f(x)
    if y * y0 < 0
      if abs(x-x1) <= xatol
        return x, true
      end
      x1, y1 = x, y
      if lastside == +1
        y0 /= 2
      end
      lastside = +1
    else
      if abs(x0-x) <= xatol
        return x, true
      end
      x0, y0 = x, y
      if lastside == -1
        y1 /= 2
      end
      lastside = -1
    end
  end
  return x, false
end


"""
    value_deriv(f, x)

Compute the value and derivative `f(x)` using ForwardDiff.jl.
"""
function value_deriv(f, x::T) where {T}
    tag = typeof(ForwardDiff.Tag(f, T))
    y = f(ForwardDiff.Dual{tag}(x,one(x)))
    ForwardDiff.value(tag, y), ForwardDiff.partials(tag, y, 1)
end

function find_zero(f::F, x0::T, ::NewtonsMethod,
                   xatol=1e-3,
                   maxiters=10_000) where {F, T<:AbstractFloat}
  for i in 1:maxiters
    y,y′ = value_deriv(f, x0)
    x1 = x0 - y/y′
    if abs(x0-x1) <= xatol
      return x1, true
    end
    x0 = x1
  end
  return x0, false
end

end
