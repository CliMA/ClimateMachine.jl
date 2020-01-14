"""
    RootSolvers

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

export find_zero, SecantMethod, RegulaFalsiMethod, NewtonsMethodAD, NewtonsMethod
export CompactSolution, VerboseSolution

import ForwardDiff

abstract type RootSolvingMethod end
Base.broadcastable(method::RootSolvingMethod) = Ref(method)

struct SecantMethod <: RootSolvingMethod end
struct RegulaFalsiMethod <: RootSolvingMethod end
struct NewtonsMethodAD <: RootSolvingMethod end
struct NewtonsMethod <: RootSolvingMethod end

abstract type SolutionType end
Base.broadcastable(soltype::SolutionType) = Ref(soltype)

"""
    VerboseSolution <: SolutionType

Used to return a [`VerboseSolutionResults`](@ref)
"""
struct VerboseSolution <: SolutionType end

abstract type AbstractSolutionResults{AbstractFloat} end

struct VerboseSolutionResults{FT} <: AbstractSolutionResults{FT}
  "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
  root::FT
  "indicates convergence"
  converged::Bool
  "error of the root of the equation ``f(x^*) = 0``"
  err::FT
  "number of iterations performed"
  iter_performed::Int
  "solution per iteration"
  root_history::Vector{FT}
  "error of the root of the equation ``f(x^*) = 0`` per iteration"
  err_history::Vector{FT}
end
SolutionResults(soltype::VerboseSolution, args...) = VerboseSolutionResults(args...)


"""
    CompactSolution <: SolutionType

Used to return a [`CompactSolutionResults`](@ref)
"""
struct CompactSolution <: SolutionType end

struct CompactSolutionResults{FT} <: AbstractSolutionResults{FT}
  "solution ``x^*`` of the root of the equation ``f(x^*) = 0``"
  root::FT
  "indicates convergence"
  converged::Bool
end
SolutionResults(soltype::CompactSolution, root, converged, args...) = CompactSolutionResults(root, converged)

init_history(::VerboseSolution, x::FT) where {FT<:AbstractFloat} = FT[x]
init_history(::CompactSolution, x) = nothing
init_history(::VerboseSolution, ::Type{FT}) where {FT<:AbstractFloat} = FT[]
init_history(::CompactSolution, ::Type{FT}) where {FT<:AbstractFloat} = nothing

push_history!(history::Vector{FT}, x::FT, ::VerboseSolution) where {FT<:AbstractFloat} =
  push!(history, x)
push_history!(history::Nothing, x::FT, ::CompactSolution) where {FT<:AbstractFloat} =
  nothing


# TODO: CuArrays.jl has trouble with isapprox on 1.1
# we use simple checks for now, will switch to relative checks later.

"""
    sol = find_zero(f[, f′], x0[, x1], method, solutiontype,
                    xatol=1e-3,
                    maxiters=10_000)

Finds the nearest root of `f` to `x0` and `x1`. Returns a the value of the root `x` such
that `f(x) ≈ 0`, and a Boolean value `converged` indicating convergence.

`method` can be one of:
- `SecantMethod()`: [Secant method](https://en.wikipedia.org/wiki/Secant_method)
- `RegulaFalsiMethod()`: [Regula Falsi method](https://en.wikipedia.org/wiki/False_position_method#The_regula_falsi_(false_position)_method).
- `NewtonsMethodAD()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) using Automatic Differentiation
  - The `x1` argument is omitted for Newton's method.
- `NewtonsMethod()`: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
  - The `x1` argument is omitted for Newton's method.
  - `f′`: derivative of function `f` whose zero is sought

The keyword arguments:
- `xatol` is the absolute tolerance of the input.
- `maxiters` is the maximum number of iterations.
"""
function find_zero end

function find_zero(f::F, x0::FT, x1::FT, ::SecantMethod, soltype::SolutionType,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  x_history = init_history(soltype, x0)
  y_history = init_history(soltype, y0)
  for i in 1:maxiters
    Δx = x1 - x0
    Δy = y1 - y0
    x0, y0 = x1, y1
    push_history!(x_history, x0, soltype)
    push_history!(y_history, y0, soltype)
    x1 -= y1 * Δx / Δy
    y1 = f(x1)
    if abs(x0-x1) <= xatol
      return SolutionResults(soltype, x1, true, y1, i, x_history, y_history)
    end
  end
  return SolutionResults(soltype, x1, false, y1, maxiters, x_history, y_history)
end

function find_zero(f::F, x0::FT, x1::FT, ::RegulaFalsiMethod, soltype::SolutionType,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  y0 = f(x0)
  y1 = f(x1)
  @assert y0 * y1 < 0
  x_history = init_history(soltype, x0)
  y_history = init_history(soltype, y0)
  lastside = 0
  local x, y
  for i in 1:maxiters
    x = (x0 * y1 - x1 * y0)/ (y1 - y0)
    y = f(x)
    push_history!(x_history, x, soltype)
    push_history!(y_history, y, soltype)
    if y * y0 < 0
      if abs(x-x1) <= xatol
        return SolutionResults(soltype, x, true, y, i, x_history, y_history)
      end
      x1, y1 = x, y
      if lastside == +1
        y0 /= 2
      end
      lastside = +1
    else
      if abs(x0-x) <= xatol
        return SolutionResults(soltype, x, true, y, i, x_history, y_history)
      end
      x0, y0 = x, y
      if lastside == -1
        y1 /= 2
      end
      lastside = -1
    end
  end
  return SolutionResults(soltype, x, false, y, maxiters, x_history, y_history)
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

function find_zero(f::F, x0::FT, ::NewtonsMethodAD, soltype::SolutionType,
                   xatol=FT(1e-3),
                   maxiters=10_000) where {F, FT<:AbstractFloat}
  local y
  x_history = init_history(soltype, FT)
  y_history = init_history(soltype, FT)
  if soltype isa VerboseSolution
    y,y′ = value_deriv(f, x0)
    push_history!(x_history, x0, soltype)
    push_history!(y_history, y , soltype)
  end
  for i in 1:maxiters
    y,y′ = value_deriv(f, x0)
    x1 = x0 - y/y′
    push_history!(x_history, x1, soltype)
    push_history!(y_history, y,  soltype)
    if abs(x0-x1) <= xatol
      return SolutionResults(soltype, x1, true, y, i, x_history, y_history)
    end
    x0 = x1
  end
  return SolutionResults(soltype, x0, false, y, maxiters, x_history, y_history)
end

function find_zero(f::F, f′::F′, x0::FT, ::NewtonsMethod, soltype::SolutionType,
                   xatol=1e-3,
                   maxiters=10_000) where {F, F′, FT<:AbstractFloat}
  x_history = init_history(soltype, FT)
  y_history = init_history(soltype, FT)
  if soltype isa VerboseSolution
    y,y′ = f(x0), f′(x0)
    push_history!(x_history, x0, soltype)
    push_history!(y_history, y , soltype)
  end
  for i in 1:maxiters
    y,y′ = f(x0), f′(x0)
    x1 = x0 - y/y′
    push_history!(x_history, x1, soltype)
    push_history!(y_history, y,  soltype)
    if abs(x0-x1) <= xatol
      return SolutionResults(soltype, x1, true, y, i, x_history, y_history)
    end
    x0 = x1
  end
  return SolutionResults(soltype, x0, false, y, maxiters, x_history, y_history)
end

end
