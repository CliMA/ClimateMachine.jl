# Contribution Guide for Abstract Time-stepping Algorithms

This guide gives a brief overview on how time-stepping methods are
implemented in [CLIMA](https://github.com/climate-machine), and
how one might contribute a new time-stepping method.

Currently, CLIMA supports a variety of time-stepping methods within
the Runge-Kutta framework. For purely explicit time-integration,
CLIMA supports the following methods:
1. [LSRK54CarpenterKennedy](@ref LSRK54CarpenterKennedy)
2. [LSRK144NiegemannDiehlBusch](@ref LSRK144NiegemannDiehlBusch)
3. [SSPRK33ShuOsher](@ref SSPRK33ShuOsher)
4. [SSPRK34SpiteriRuuth](@ref SSPRK34SpiteriRuuth)

Methods 1 and 2 are implemented as low-storage Runge-Kutta methods,
which uses a 2N storage scheme for the coefficient arrays of
the given time-stepping method (known as the Butcher Tableau). All
time-integration methods are part of a single **module**: **ODESolvers**.
Each Runge-Kutta method requires **one struct**, with a **constructor**.

## Basic Template for an explicit Runge-Kutta Method

A basic template for an explicit Runge-Kutta method is
as follows:

```julia
export MyExplicitRKMethod

struct MyExplicitRKMethod{T, RT, AT, Nstages} <: AbstractODESolver
    "time step size"
    dt::RT
    "rhs function"
    rhs!
    "Storage for the stage vectors"
    Qstage::AT
    "RK coefficient vector A (rhs scaling)"
    RKA::Array{RT, 2}
    "RK coefficient vector B (rhs accumulation scaling)"
    RKB::Array{RT, 1}
    "RK coefficient vector C (temporal scaling)"
    RKC::Array{RT, 1}
    # May require more attributes depending on the type of RK method

    # Constructor
    function MyExplicitRKMethod(args...)
        # Body of the constructor
        ...
        return MyExplicitRKMethod(dt, rhs, Qstage, RKA, RKB, RKC)
    end
end
```

Once `MyExplicitRKMethod` is defined, we require to implement
an appropriate `dostep!` function, which defines how to
step the state vector `Q` forward in time:
```julia
function ODESolver.dostep!(Q, rkmethod::MyExplicitRKMethod, p,
                           time::Real,...)
    # Function body
end
```
Once `dostep!` is implemented, `MyExplicitRKMethod` should be
ready for use in CLIMA.

## Basic Template for an IMEX/Additive Runge-Kutta Method

IMEX, or IMplicit-EXplicit, Runge-Kutta methods require a bit more
attention. IMEX methods are typically constructed from
Additively-partitioned Runge-Kutta (ARK) methods. For IMEX methods,
the standard way is to consider an ARK method with **two** partitions:
one explicit part, and one implicit part. The implicit part will require
a linear solver.

An ARK method with an explicit and implicit component will require **two***
Butcher Tableaus: one for each of the partitioned components.
Additionally, a linear solver is required.
Currently, CLIMA supports the follow set of ARK methods for IMEX-based
timestepping:
1. [ARK2GiraldoKellyConstantinescu](@ref ARK2GiraldoKellyConstantinescu)
2. [ARK548L2SA2KennedyCarpenter](@ref ARK548L2SA2KennedyCarpenter)
3. [ARK437L2SA1KennedyCarpenter](@ref ARK437L2SA1KennedyCarpenter)

For example, consider the following:

```julia
export MyIMEXMethod

using ..LinearSolvers
const LS = LinearSolvers

struct MyIMEXMethod{T, RT, AT, LT, V, VS, Nstages, Nstages_sq} <: AbstractODESolver
    "time step"
    dt::RT
    "rhs function"
    rhs!
    "rhs linear operator"
    rhs_linear!
    "implicit operator, pre-factorized"
    implicitoperator!
    "linear solver"
    linearsolver::LT
    "Stage vectors for the ARK method"
    Qstages::NTuple{Nstages, AT}
    "RK coefficient matrix A for the explicit scheme"
    RKA_explicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient matrix A for the implicit scheme"
    RKA_implicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient vector B (rhs accumulation scaling)"
    RKB::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector C (temporal scaling)"
    RKC::SArray{Tuple{Nstages}, RT, 1, Nstages}

    # May have more attributes depending on the method

    # Constructor
    function MyIMEXMethod(args...)
        # Body of the constructor
        ...
        return MyIMEXMethod(dt, rhs, rhs_linear, implicitoperator,
                            Qstages, RKA_explicit, RKA_implicit,
                            RKB, RKC)
    end
```

In addition to a `dostep!` function, IMEX methods also require
functions related to the `implicitoperator`, which should be interpreted
as a matrix operator representing the implicit components. Depending on
the coefficients in `RKA_implicit`, a linear solve may be required at each
stage of the ARK method, or only a subset of the total stages. If the implicit
operator is changing with each stage, then it will need to be updated
via a **function** `updatedt!`:
```julia
function updatedt!(ark::MyIMEXMethod, dt::Real)
    # Function body
    ark.implicitoperator! = prefactorize(...)
end
```
For information on the function `prefactorize`, see
the **module** `CLIMA.LinearSolvers`.
