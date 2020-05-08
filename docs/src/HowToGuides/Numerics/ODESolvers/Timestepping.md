# Contribution Guide for Abstract Time-stepping Algorithms

This guide gives a brief overview on how time-stepping methods are
implemented in [ClimateMachine](https://github.com/CliMA/ClimateMachine.jl),
and how one might contribute a new time-stepping method.

Currently, ClimateMachine supports a variety of time-stepping methods
within the Runge-Kutta framework. For purely explicit time-integration,
ClimateMachine supports the following methods:

1. [LSRK54CarpenterKennedy](@ref LSRK54CarpenterKennedy)
2. [LSRK144NiegemannDiehlBusch](@ref LSRK144NiegemannDiehlBusch)
3. [SSPRK33ShuOsher](@ref SSPRK33ShuOsher)
4. [SSPRK34SpiteriRuuth](@ref SSPRK34SpiteriRuuth)

Methods 1 and 2 are implemented as low-storage Runge-Kutta methods,
which uses a 2N storage scheme for the coefficient arrays of the given
time-stepping method (known as the Butcher Tableau). All time-integration
methods are part of a single **module**: **ODESolvers**.  Each Runge-Kutta
method requires **one struct**, with a **constructor**.

## Basic Template for an explicit Runge-Kutta Method

A basic template for an explicit Runge-Kutta method is as follows:

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

Once `MyExplicitRKMethod` is defined, we require to implement an appropriate
`dostep!` function, which defines how to step the state vector `Q` forward
in time:

```julia
function ODESolver.dostep!(Q, rkmethod::MyExplicitRKMethod, p,
                           time::Real,...)
    # Function body
end
```
Once `dostep!` is implemented, `MyExplicitRKMethod` should be ready for
use in ClimateMachine.

## Basic Template for an IMEX/Additive Runge-Kutta Method

IMEX, or IMplicit-EXplicit, Runge-Kutta methods require a bit more
attention. IMEX methods are typically constructed from Additively-partitioned
Runge-Kutta (ARK) methods. For IMEX methods, the standard way is to consider
an ARK method with **two** partitions: one explicit part, and one implicit
part. The implicit part will require a linear solver.


An ARK method with an explicit and implicit component will require **two**
Butcher Tableaus: one for each of the partitioned components.  Additionally,
a linear solver is required.  Currently, ClimateMachine supports the follow
set of ARK methods for IMEX-based timestepping:

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

In addition to a `dostep!` function, IMEX methods also require functions
related to the `implicitoperator`, which should be interpreted as a matrix
operator representing the implicit components. Depending on the coefficients
in `RKA_implicit`, a linear solve may be required at each stage of the ARK
method, or only a subset of the total stages. If the implicit operator is
changing with each stage, then it will need to be updated via a **function**
`updatedt!`:

```julia
function updatedt!(ark::MyIMEXMethod, dt::Real)
    # Function body
    ark.implicitoperator! = prefactorize(...)
end
```
For information on the function `prefactorize`, see
the **module** `ClimateMachine.LinearSolvers`.

## The Struct and its Constructor

The `Struct` defining important quantities for a given time-integrator is
a subset of an `AbstractODESolver`. For simplicity, we assume a standard
Runge-Kutta method:

```julia
struct MyRKMethod{T, RT, AT, Nstages} <: AbstractODESolver
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
    function MyRKMethod(args...)
        # Body of the constructor
        ...
        return MyRKMethod(constructor_args...)
    end
end
```
Since time-integration methods are often complex and drastically different
from one another, the `Struct` and its `Constructor`, `MyRKMethod(args...)`,
will often look quite different, i.e. explicit and IMEX time-integrators
have different `Struct` attributes and `Constructor` arguments.

As a general rule of thumb, all Runge-Kutta-based methods will need
to keep track of the time-step size `dt` as wells as the Butcher
tableau coefficients. If your time-integrator has an implicit component
(semi-implicit) or is fully implicit, the `Struct` will need to know about
the `implicitoperator` and the corresponding `linearsolver`.

## The `dostep!` function

No matter the type of time-integration method, **all** time-steppers
require the implementation of the `dostep!` function. Suppose we have some
time-stepper, say `MyTimeIntegrator`. Then the arguments to the `dostep!`
function will be:

```julia
function dostep!(
    Q,
    rkmethod::MyTimeIntegrator,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    in_slow_scaling = nothing,
)
    # Function body
end
```
Where `Q` is the state vector, `time` denotes the time for the
next time-step, the time-integrator, and `slow_δ`, `slow_rv_dQ`,
`in_slow_scaling` are optional arguments contributing to additional terms
in the ODE right-hand side. More information on those argument will be
covered in a later section. Note that the argument `p` should be interpreted
as a context manager for more sophisticated time-stepping methods (for
example, schemes with *multiple* RK methods); typical Runge-Kutta schemes
will generally not need to worry about the argument `p`.  The argument
`rkmethod` is used for multiple dispatch, and `Q` is an array that gets
overwritten with field values at the next time-step.

## Multirate Runge-Kutta Methods

Multirate time-integration is a popular approach for weather and climate
simulations. The core idea is that the ODE in question can be expressed
as the sum of a `fast` and `slow` component. In the atmosphere, `fast`
dynamics include the modes associated with acoustic waves (assuming a
compressible or pseudo-compressible model of the atmosphere), typically on
the order of 300 m/s, while dynamics associated with advection, diffusion,
and radiation represent `slow` dynamics. The core idea behind a multirate
method is to step each component (fast and slow) forward in time at a
different rate (hence the name "Multi-Rate").

There are several different approaches for multirate
methods. In ClimateMachine, a multirate time-stepper is provided as
[MultirateRungeKutta](@ref MultirateRungeKutta), which takes a given number
of Runge-Kutta methods (one for each rate).

### Implementation Considerations
Generally speaking, a multirate method requires composing several different
time-stepping methods for different components of the ODE. Therefore, the
`Struct` and its `Constructor` may look something like:

```julia
export MyMultirateMethod

struct MyMultirateMethod{SS, FS, RT} <: AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    # May require more attributes depending on implementation

    # Constructor
    function MyMultirateMethod(args...)
        # Body of constructor
        ...
        return MyMultirateMethod(constructor_args...)
    end
end
```

One can imagine a scenario where several rates are operating in tandem. There
are a number of possible approaches for handling this. One example is to
*recursively* nest multiple `MyMultirateMethod` instances:

```julia
function MyMultirateMethod(solvers::Tuple, Q; dt::Real)
    # Take a tuple of solvers and defined a nested
    # multirate method
    fast_solver = MyMultirateMethod(solvers[2:end], Q; dt = dt)
    slow_solver = solver[1]
    return MyMultirateMethod(slow_solver, fast_solver, Q; dt = dt)
end
```
Note that this example assumes the solvers `Tuple` is ordered in such a
way that the first element is the *slowest* solver, while all subsequent
solvers are faster than the previous.

Just like all other previously mentioned time-integrators, the `dostep!`
function will need to be implemented, taking into account the nesting of
several solvers.

## Writing Tests

Testing is critical for the success and sustainability of any software
project. Therefore, it is absolutely *imperative* for all newly added
time-integrator to have a corresponding regression test.

The standard way is to consider an ODE with an analytic solution.
A given time-integrator will have a known convergence rate, and thus
a good regression test would be to verify temporal convergence
in the computed solution. Several examples can be found in
`ClimateMachine.jl/test/ODESolvers`.


## Performance Checks

Timing performance of a time-integrator can be done using standard guidelines
for CPU and GPU architectures. Certain factors that impact the performance
of a time-integrator includes the following:

1. Memory management -- how much memory is a given method using, in
particular, storing stage vectors for RK methods. For IMEX methods, using
direct solvers (LU factorization, for example) often has a significant
impact on memory usage.

2. Right-hand side evaluations -- for explicit methods, the total number
of function evaluations contributes to most of the arithmetic intensity
of the time-integrator. More evaluates require more compute resources.

3. Solving linear systems -- for IMEX or implicit methods, solving a linear
system of equations is required. This is arguably the most expensive
part of any IMEX/implicit time-integrator. Things to consider include:
iterative methods, preconditioning, and parallel scalability.

