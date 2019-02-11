"""
# RootSolvers

Module containing functions for solving roots of non-linear
equations. The returned result is a tuple of the root and
a Bool indicating convergence.

```
find_zero(f::F,
           x_0::T,
           x_1::T,
           args::Tuple,
           iter_params::IterParams{R, Int},
           method::RootSolvingMethod
           )::Tuple{T, Bool} where {F, R, T <: Union{R, AbstractArray{R}}}
```

---

## Interface
- [`find_zero`](@ref) compute x^* such that f(x^*) = 0

## Arguments
- `f` equation roots function, where `f` is callable via `f(x, args...)`
- `x_0, x_1` initial guesses
- [`IterParams`](@ref) struct containing absolute tolerance on f(x^*) and maximum iterations
- [`RootSolvingMethod`](@ref) Algorithm to solve roots of the equation:
  - [`SecantMethod`](@ref) Secant method
  - [`RegulaFalsiMethod`](@ref) Regula Falsi Method

## Single example
```
julia> using RootSolvers
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
```

## Broadcast example

To broadcast, wrap arguments that need not be broadcasted in a Ref.

```
julia> using RootSolvers
x_0 = rand(5,5)
x_1 = rand(5,5).+2.0
f(x, y) = x^2 - y
x_star2 = 10000.0
args = Tuple(x_star2)
x_star = sqrt(x_star2)
tol_abs = 1.0e-3
iter_max = 100
x_root, converged = find_zero.(f,
                               x_0,
                               x_1,
                               Ref(args),
                               Ref(IterParams(tol_abs, iter_max)),
                               Ref(SecantMethod()))
```

"""
module RootSolvers

export find_zero
export IterParams

export SecantMethod
export RegulaFalsiMethod

struct IterParams{R<:AbstractFloat, I<:Int}
  tol_abs::R
  iter_max::I
end

abstract type RootSolvingMethod end
struct SecantMethod<:RootSolvingMethod end
struct RegulaFalsiMethod<:RootSolvingMethod end

get_solve_methods() = (SecantMethod(), RegulaFalsiMethod())

"""
`find_zero(f, x_0, x_1, args, iter_params, SecantMethod)`

Solves the root equation, `f`, using Secant method.
See [RootSolvers](@ref) for more information.
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

"""
`find_zero(f, x_0, x_1, args, iter_params, RegulaFalsiMethod)`

Solves the root equation, `f`, using Regula Falsi method.
See [RootSolvers](@ref) for more information.
"""
function find_zero(f::F,
                   x_0::T,
                   x_1::T,
                   args::Tuple,
                   iter_params::IterParams{R, Int},
                   method::RegulaFalsiMethod
                   )::Tuple{T, Bool} where {F, R, T <: Union{R, AbstractArray{R}}}
  iter_max = iter_params.iter_max
  tol_abs = iter_params.tol_abs
  a = x_0
  b = x_1
  fa = f(a, args...)
  fb = f(b, args...)
  c = a
  i = 0
  for i in 1:iter_max
    fa = f(a, args...)
    fb = f(b, args...)
    c = (a * fb - b * fa)/ (fb - fa)
    fc = f(c, args...)
    if abs(fc) < tol_abs
      return c, true
    elseif fc * fa < 0
      b = c
    else
      a = c
    end
  end
  return c, false
end

end