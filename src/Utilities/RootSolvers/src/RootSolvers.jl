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

export find_zero, SecantMethod, RegulaFalsiMethod

abstract type RootSolvingMethod end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

struct SecantMethod<:RootSolvingMethod end
struct RegulaFalsiMethod<:RootSolvingMethod end

"""
    x, converged = find_zero(f, x0, x1, method;
                             xatol=0, xrtol=sqrt(eps(eltype(x0))), 
                             yatol=sqrt(eps(eltype(x0))), maxiters=10_000)

Finds the nearest root of `f` to `x0` and `x1`. Returns a the value of the root `x` such
that `f(x) ≈ 0`, and a Boolean value `converged` indicating convergence.

`method` can be one of:
- `SecantMethod()`: [Secant method](https://en.wikipedia.org/wiki/Secant_method)
- `RegulaFalsiMethod()`: [Regula Falsi method](https://en.wikipedia.org/wiki/False_position_method#The_regula_falsi_(false_position)_method).

The keyword arguments:
- `xatol` and `xrtol` are the absolute and relative tolerances of the input.
- `yatol` is the absolute tolerance of the output of `f`.
- `maxiters` is the maximum number of iterations.

"""
function find_zero end


function find_zero(f::F, x0::T, x1::T, ::SecantMethod;
                   xatol=zero(x0), xrtol=sqrt(eps(T)), yatol=sqrt(eps(T)), maxiters=10_000) where {F, T<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  for i in 1:maxiters
    Δx = x1 - x0
    Δy = y1 - y0
    x0, y0 = x1, y1
    x1 -= y1 * Δx / Δy
    y1 = f(x1)
    if isapprox(y1,0; atol=yatol) || isapprox(x0,x1; atol=xatol, rtol=xrtol)
      return x1, true
    end
  end
  return x1, false
end

function find_zero(f::F, x0::T, x1::T, ::RegulaFalsiMethod;
                   xatol=zero(x0), xrtol=sqrt(eps(T)), yatol=sqrt(eps(T)), maxiters=10_000) where {F, T<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  @assert y0 * y1 < 0
  lastside = 0
  local x
  for i in 1:maxiters
    x = (x0 * y1 - x1 * y0)/ (y1 - y0)
    y = f(x)
    if y * y0 < 0
      if isapprox(x,x1; atol=xatol, rtol=xrtol)
        return x, true
      end
      x1, y1 = x, y
      if lastside == +1
        y0 /= 2
      end
      lastside = +1
    else
      x0, y0 = x, y
      if lastside == -1
        y1 /= 2
      end
      lastside = -1
    end
    if isapprox(y,0; atol=yatol) || isapprox(x0,x1; atol=xatol, rtol=xrtol)
      return x, true
    end
  end
  return x, false
end

end
