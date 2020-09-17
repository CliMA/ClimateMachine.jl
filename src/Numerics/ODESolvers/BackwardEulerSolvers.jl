export LinearBackwardEulerSolver, AbstractBackwardEulerSolver
export NonLinearBackwardEulerSolver

abstract type AbstractImplicitOperator end

"""
    op! = EulerOperator(f!, ϵ)

Construct a linear operator which performs an explicit Euler step ``Q + α
f(Q)``, where `f!` and `op!` both operate inplace, with extra arguments passed
through, i.e.
```
op!(LQ, Q, args...)
```
is equivalent to
```
f!(dQ, Q, args...)
LQ .= Q .+ ϵ .* dQ
```
"""
mutable struct EulerOperator{F, FT} <: AbstractImplicitOperator
    f!::F
    ϵ::FT
end

function (op::EulerOperator)(LQ, Q, args...)
    op.f!(LQ, Q, args..., increment = false)
    @. LQ = Q + op.ϵ * LQ
end

"""
    AbstractBackwardEulerSolver

An abstract backward Euler method
"""
abstract type AbstractBackwardEulerSolver end

"""
    (be::AbstractBackwardEulerSolver)(Q, Qhat, α, param, time)

Each concrete implementations of `AbstractBackwardEulerSolver` should provide a
callable version which solves the following system for `Q`
```
    Q = Qhat + α f(Q, param, time)
```
where `f` is the ODE tendency function, `param` are the ODE parameters, and
`time` is the current ODE time. The arguments `Q` should be modified in place
and should not be assumed to be initialized to any value.
"""
(be::AbstractBackwardEulerSolver)(Q, Qhat, α, p, t) =
    throw(MethodError(be, (Q, Qhat, α, p, t)))

"""
    Δt_is_adjustable(::AbstractBackwardEulerSolver)

Return `Bool` for whether this backward Euler solver can be updated. default is
`false`.
"""
Δt_is_adjustable(::AbstractBackwardEulerSolver) = false

"""
    update_backward_Euler_solver!(::AbstractBackwardEulerSolver, α)

Update the given backward Euler solver for the parameter `α`; see
['AbstractBackwardEulerSolver'](@ref). Default behavior is no change to the
solver.
"""
update_backward_Euler_solver!(::AbstractBackwardEulerSolver, Q, α) = nothing

"""
    setup_backward_Euler_solver(solver, Q, α, tendency!)

Returns a concrete implementation of an `AbstractBackwardEulerSolver` that will
solve for `Q` in systems of the form of
```
    Q = Qhat + α f(Q, param, time)
```
where `tendency!` is the in-place tendency function. Not the array `Q` is just
passed in for type information, e.g., `Q` the same `Q` will not be used for all
calls to the solver.
"""
setup_backward_Euler_solver(solver::AbstractBackwardEulerSolver, _...) = solver

"""
    LinearBackwardEulerSolver(::AbstractSystemSolver; isadjustable = false)

Helper type for specifying building a backward Euler solver with a linear
solver.  If `isadjustable == true` then the solver can be updated with a new
time step size.
"""
struct LinearBackwardEulerSolver{LS}
    solver::LS
    isadjustable::Bool
    preconditioner_update_freq::Int
    LinearBackwardEulerSolver(
        solver;
        isadjustable = false,
        preconditioner_update_freq = -1,
    ) = new{typeof(solver)}(solver, isadjustable, preconditioner_update_freq)
end

"""
    LinBESolver

Concrete implementation of an `AbstractBackwardEulerSolver` to use linear
solvers of type `AbstractSystemSolver`. See helper type
[`LinearBackwardEulerSolver`](@ref)
```
    Q = Qhat + α f(Q, param, time)
```
"""
mutable struct LinBESolver{FT, F, LS} <: AbstractBackwardEulerSolver
    α::FT
    f_imp!::F
    solver::LS
    isadjustable::Bool
    # used only for iterative solver
    preconditioner::AbstractPreconditioner
    # used only for direct solver
    factors
end

Δt_is_adjustable(lin::LinBESolver) = lin.isadjustable

function setup_backward_Euler_solver(
    lin::LinearBackwardEulerSolver,
    Q,
    α,
    f_imp!,
)
    FT = eltype(α)
    rhs! = EulerOperator(f_imp!, -α)

    factors = prefactorize(rhs!, lin.solver, Q, nothing, FT(NaN))

    # when direct solver is applied preconditioner_update_freq <= 0
    @assert(
        typeof(lin.solver) <: AbstractIterativeSystemSolver ||
        lin.preconditioner_update_freq <= 0
    )

    preconditioner_update_freq = lin.preconditioner_update_freq
    # construct an empty preconditioner
    preconditioner = (
        preconditioner_update_freq > 0 ?
            ColumnwiseLUPreconditioner(f_imp!, Q, preconditioner_update_freq) :
            NoPreconditioner()
    )

    LinBESolver(
        α,
        f_imp!,
        lin.solver,
        lin.isadjustable,
        preconditioner,
        factors,
    )
end

function update_backward_Euler_solver!(lin::LinBESolver, Q, α)
    lin.α = α
    FT = eltype(Q)
    # for direct solver, update factors
    # for iterative solver, set factors to Nothing (TODO optimize)
    lin.factors = prefactorize(
        EulerOperator(lin.f_imp!, -α),
        lin.solver,
        Q,
        nothing,
        FT(NaN),
    )
end

function (lin::LinBESolver)(Q, Qhat, α, p, t)
    rhs! = EulerOperator(lin.f_imp!, -α)

    if lin.α != α
        @assert lin.isadjustable
        update_backward_Euler_solver!(lin, Q, α)
    end

    if typeof(lin.solver) <: AbstractIterativeSystemSolver
        FT = eltype(α)
        preconditioner_update!(
            rhs!,
            rhs!.f!,
            lin.preconditioner,
            nothing,
            FT(NaN),
        )
        linearsolve!(rhs!, lin.preconditioner, lin.solver, Q, Qhat, p, t)
        preconditioner_counter_update!(lin.preconditioner)
    else
        linearsolve!(rhs!, lin.factors, lin.solver, Q, Qhat, p, t)
    end
end

"""
    struct NonLinearBackwardEulerSolver{NLS}
        nlsolver::NLS
        isadjustable::Bool
        preconditioner_update_freq::Int64
    end

Helper type for specifying building a nonlinear backward Euler solver with a nonlinear
solver.

# Arguments
- `nlsolver`: iterative nonlinear solver, i.e., JacobianFreeNewtonKrylovSolver
- `isadjustable`: TODO not used, might use for updating preconditioner
- `preconditioner_update_freq`:  relavent to Jacobian free -1: no preconditioner;
                             positive number, update every freq times
"""
struct NonLinearBackwardEulerSolver{NLS}
    nlsolver::NLS
    isadjustable::Bool
    # preconditioner_update_freq, -1: no preconditioner;
    # positive number, update every freq times
    preconditioner_update_freq::Int
    function NonLinearBackwardEulerSolver(
        nlsolver;
        isadjustable = false,
        preconditioner_update_freq = -1,
    )
        NLS = typeof(nlsolver)
        return new{NLS}(nlsolver, isadjustable, preconditioner_update_freq)
    end
end


"""
    LinBESolver

Concrete implementation of an `AbstractBackwardEulerSolver` to use nonlinear
solvers of type `NLS`. See helper type
[`NonLinearBackwardEulerSolver`](@ref)
```
    Q = Qhat + α f_imp(Q, param, time)
```
"""
mutable struct NonLinBESolver{FT, F, NLS} <: AbstractBackwardEulerSolver
    # Solve Q - α f_imp(Q) = Qrhs, not used
    α::FT
    # implcit operator
    f_imp!::F
    # jacobian action, which approximates drhs!/dQ⋅ΔQ , here rhs!(Q) = Q - α f_imp(Q)
    jvp!::JacobianAction
    # nonlinear solver
    nlsolver::NLS
    # whether adjust the time step or not, not used
    isadjustable::Bool
    # preconditioner, approximation of drhs!/dQ
    preconditioner::AbstractPreconditioner

end

Δt_is_adjustable(nlsolver::NonLinBESolver) = nlsolver.isadjustable

"""
    setup_backward_Euler_solver(solver::NonLinearBackwardEulerSolver, Q, α, tendency!)

Returns a concrete implementation of an `AbstractBackwardEulerSolver` that will
solve for `Q` in nonlinear systems of the form of
```
    Q = Qhat + α f(Q, param, time)
```
Create an empty JacobianAction

Create an empty preconditioner if preconditioner_update_freq > 0
"""
function setup_backward_Euler_solver(
    nlbesolver::NonLinearBackwardEulerSolver,
    Q,
    α,
    f_imp!,
)
    # Create an empty JacobianAction (without operator)
    jvp! = JacobianAction(nothing, Q, nlbesolver.nlsolver.ϵ)

    # Create an empty preconditioner if preconditioner_update_freq > 0
    preconditioner_update_freq = nlbesolver.preconditioner_update_freq
    # construct an empty preconditioner
    preconditioner = (
        preconditioner_update_freq > 0 ?
            ColumnwiseLUPreconditioner(f_imp!, Q, preconditioner_update_freq) :
            NoPreconditioner()
    )
    NonLinBESolver(
        α,
        f_imp!,
        jvp!,
        nlbesolver.nlsolver,
        nlbesolver.isadjustable,
        preconditioner,
    )
end

"""
Nonlinear solve

Update rhs! with α

Update the rhs! in the jacobian action jvp!
"""
function (nlbesolver::NonLinBESolver)(Q, Qhat, α, p, t)

    rhs! = EulerOperator(nlbesolver.f_imp!, -α)

    nlbesolver.jvp!.rhs! = rhs!

    nonlinearsolve!(
        rhs!,
        nlbesolver.jvp!,
        nlbesolver.preconditioner,
        nlbesolver.nlsolver,
        Q,
        Qhat,
        p,
        t,
    )

end
