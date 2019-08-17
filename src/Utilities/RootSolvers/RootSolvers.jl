"""
# RootSolvers

Module containing functions for solving roots of non-linear
equations. See [`find_zero`](@ref).


## Example

```
using CLIMA.Utilities.RootSolvers

sol = find_zero(x -> x^2 - 100^2, 0.0, 1000.0, SecantMethod(), CompactSolution())
x_root = sol.root
converged = sol.converged
```

"""
module RootSolvers

export find_zero, SecantMethod, RegulaFalsiMethod, NewtonsMethod
export CompactSolution, VerboseSolution

import ForwardDiff

abstract type RootSolvingMethod end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

struct SecantMethod <: RootSolvingMethod end
struct RegulaFalsiMethod <: RootSolvingMethod end
struct NewtonsMethod <: RootSolvingMethod end

abstract type SolutionType end
Base.broadcastable(soltype::SolutionType) = Ref(soltype)
"""
    CompactSolution <: SolutionType

Used to return a root solver solution with:
 - `root`           the solution
 - `converged`      a `Bool` indicating convergence
"""
struct CompactSolution <: SolutionType end

"""
    CompactSolution <: SolutionType

Used to return a root solver solution with:
 - `root`           the solution
 - `converged`      a `Bool` indicating convergence
 - `err`            the error of the root (`f(x_sol)`)
 - `iter_performed` number of iterations performed
"""
struct VerboseSolution <: SolutionType end

abstract type AbstractSolutionResults{T} end
struct VerboseSolutionResults{T} <: AbstractSolutionResults{T}
  root::T
  converged::Bool
  err::T
  iter_performed::Int
end
function SolutionResults(root::T, converged::Bool, err::T, iter_performed::Int, ::VerboseSolution) where T
  VerboseSolutionResults{T}(root, converged, err, iter_performed)
end

struct CompactSolutionResults{T} <: AbstractSolutionResults{T}
  root::T
  converged::Bool
end
function SolutionResults(root::T, converged::Bool, err::T, iter_performed::Int, ::CompactSolution) where T
  CompactSolutionResults{T}(root, converged)
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

function find_zero(f::F, x0::T, x1::T, ::SecantMethod, soltype::SolutionType,
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
      return SolutionResults(x1, true, y1, i, soltype)
    end
  end
  return SolutionResults(x1, false, y1, maxiters, soltype)
end

function find_zero(f::F, x0::T, x1::T, ::RegulaFalsiMethod, soltype::SolutionType,
                   xatol=T(1e-3),
                   maxiters=10_000) where {F, T<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  @assert y0 * y1 < 0
  lastside = 0
  local x, y
  for i in 1:maxiters
    x = (x0 * y1 - x1 * y0)/ (y1 - y0)
    y = f(x)
    if y * y0 < 0
      if abs(x-x1) <= xatol
        return SolutionResults(x, true, y, i, soltype)
      end
      x1, y1 = x, y
      if lastside == +1
        y0 /= 2
      end
      lastside = +1
    else
      if abs(x0-x) <= xatol
        return SolutionResults(x, true, y, i, soltype)
      end
      x0, y0 = x, y
      if lastside == -1
        y1 /= 2
      end
      lastside = -1
    end
  end
  return SolutionResults(x, false, y, maxiters, soltype)
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

function find_zero(f::F, x0::T, ::NewtonsMethod, soltype::SolutionType,
                   xatol=T(1e-3),
                   maxiters=10_000) where {F, T<:AbstractFloat}
  local y
  for i in 1:maxiters
    y,y′ = value_deriv(f, x0)
    x1 = x0 - y/y′
    if abs(x0-x1) <= xatol
      return SolutionResults(x1, true, y, i, soltype)
    end
    x0 = x1
  end
  return SolutionResults(x0, false, y, maxiters, soltype)
end

end
