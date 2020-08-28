export LinearBackwardEulerSolver, AbstractBackwardEulerSolver,
       BackwardEulerODESolver
using LinearAlgebra
using ..DGMethods
using CLIMAParameters.Planet: kappa_d, R_d, T_0
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
mutable struct EulerOperator{F, FT}
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
    LinearBackwardEulerSolver(solver; isadjustable = false) =
        new{typeof(solver)}(solver, isadjustable)
end

"""
    LinBESolver

Concrete implementation of an `AbstractBackwardEulerSolver` to use linear
solvers of type `AbstractSystemSolver`. See helper type
[`LinearBackwardEulerSolver`](@ref)
"""
mutable struct LinBESolver{FT, FAC, LS, F} <: AbstractBackwardEulerSolver
    α::FT
    factors::FAC
    solver::LS
    isadjustable::Bool
    rhs!::F
end
Δt_is_adjustable(lin::LinBESolver) = lin.isadjustable

function setup_backward_Euler_solver(lin::LinearBackwardEulerSolver, Q, α, rhs!)
    FT = eltype(α)
    if isnothing(rhs!.schur_complement)
      factors =
          prefactorize(EulerOperator(rhs!, -α), lin.solver, Q, nothing, FT(NaN))
    else
      factors =
          prefactorize(EulerOperator(rhs!, -α), lin.solver, Q, α, rhs!)
    end
    LinBESolver(α, factors, lin.solver, lin.isadjustable, rhs!)
end

function update_backward_Euler_solver!(lin::LinBESolver, Q, α)
    lin.α = α
    FT = eltype(Q)
    lin.factors = prefactorize(
        EulerOperator(lin.rhs!, -α),
        lin.solver,
        Q,
        nothing,
        FT(NaN),
    )
end

function (lin::LinBESolver)(Q, Qhat, α, p, t)
    if lin.α != α
        @assert lin.isadjustable
        update_backward_Euler_solver!(lin, Q, α)
    end

    if lin.rhs! isa Function || isnothing(lin.rhs!.schur_complement)
      linearsolve!(lin.factors, lin.solver, Q, Qhat, p, t)
      #println("Normal result")
      #@views begin

      #  param_set = lin.rhs!.balance_law.atmos.param_set
      #  γ = 1 / (1 - kappa_d(param_set))
      #  coeff = R_d(param_set) * T_0(param_set) / (γ - 1)
      #  p = @. (γ - 1) * (Q.ρe - Q.ρ * (lin.rhs!.state_auxiliary.orientation[:, 1, :] - coeff))
      #  p = extrema(p)
      #  ρ  = extrema(Q[:, 1, :])
      #  ρu = extrema(Q[:, 2, :])
      #  ρv = extrema(Q[:, 3, :])
      #  ρw = extrema(Q[:, 4, :])
      #  ρe = extrema(Q[:, 5, :])
      #  Nq = 5
      #  Ne = 10
      #  ρw_sfc = extrema(Q[1:Nq*Nq, 4, 1:Ne:end])
      #end
      #@show p
      #@show ρ
      #@show ρu
      #@show ρv
      #@show ρw, ρw_sfc
      #@show ρe
      #error("hi")
    else
      dg = lin.rhs!
      init_schur_state(Q, Qhat, α, dg)
      schur_state = dg.states_schur_complement.state
      schur_rhs = dg.states_schur_complement.rhs
      
      p = extrema(dg.states_schur_complement.state[:, 1, :])
      @show p
      linearsolve!(schur_lhs!, lin.solver, schur_state, schur_rhs, α, dg)
      p = extrema(dg.states_schur_complement.state[:, 1, :])
      #linearsolve!(lin.factors, lin.solver, schur_state, schur_rhs, α, dg)
      schur_extract_state(Q, Qhat, α, dg)
      println("Schur result")
      @show p
      @show α
      @views begin
        ρ  = extrema(Q[:, 1, :])
        ρu = extrema(Q[:, 2, :])
        ρv = extrema(Q[:, 3, :])
        ρw = extrema(Q[:, 4, :])
        ρe = extrema(Q[:, 5, :])
       
        Nq = 5
        Ne = 10
        ρw_sfc = extrema(Q[1:Nq*Nq, 4, 1:Ne:end])
      end
      @show ρ
      @show ρu
      @show ρv
      @show ρw, ρw_sfc
      @show ρe
      #error("hi")
    end
end


mutable struct BackwardEulerODESolver{T, RT, AT, BE} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    rhs!
    "Storage for RHS"
    Qhat::AT
    "backward Euler solver"
    besolver!::BE

    function BackwardEulerODESolver(
        rhs!,
        Q::AT,
        backward_euler_solver;
        dt = 0,
        t0 = 0
    ) where {AT <: AbstractArray}
        T = eltype(Q)
        RT = real(T)

        besolver! = setup_backward_Euler_solver(
            backward_euler_solver,
            Q,
            dt,
            rhs!,
        )
        BE = typeof(besolver!)

        Qhat = similar(Q)
        fill!(Qhat, 0)

        new{T, RT, AT, BE}(RT(dt), RT(t0), rhs!, Qhat, besolver!)
    end
end

# this will only work for iterative solves
# direct solvers use prefactorization
function updatedt!(bes::BackwardEulerODESolver, dt)
    @assert Δt_is_adjustable(bes.besolver!)
    update_backward_Euler_solver!(bes.besolver!, bes.Qhat, dt)
end

function dostep!(
    Q,
    bes::BackwardEulerODESolver,
    p,
    time
)
    bes.Qhat .= Q
    bes.besolver!(Q, bes.Qhat, bes.dt, p, time)
end

