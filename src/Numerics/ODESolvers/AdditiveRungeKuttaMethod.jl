export AbstractAdditiveRungeKutta
export LowStorageVariant, NaiveVariant
export AdditiveRungeKutta
export ARK1ForwardBackwardEuler
export ARK2ImplicitExplicitMidpoint
export ARK2GiraldoKellyConstantinescu
export ARK548L2SA2KennedyCarpenter, ARK437L2SA1KennedyCarpenter
export Trap2LockWoodWeller

# Naive formulation that uses equation 3.8 from Giraldo, Kelly, and
# Constantinescu (2013) directly.  Seems to cut the number of solver iterations
# by half but requires Nstages - 1 additional storage.
struct NaiveVariant end
additional_storage(::NaiveVariant, Q, Nstages) =
    (Lstages = ntuple(i -> similar(Q), Nstages),)

# Formulation that does things exactly as in Giraldo, Kelly, and Constantinescu
# (2013).  Uses only one additional vector of storage regardless of the number
# of stages.
struct LowStorageVariant end
additional_storage(::LowStorageVariant, Q, Nstages) = (Qtt = similar(Q),)

abstract type AbstractAdditiveRungeKutta <: AbstractODESolver end

"""
    AdditiveRungeKutta(f, l, backward_euler_solver, RKAe, RKAi, RKB, RKC, Q;
                       split_explicit_implicit, variant, dt, t0 = 0)

This is a time stepping object for implicit-explicit time stepping of a
decomposed differential equation. When `split_explicit_implicit == false`
the equation is assumed to be decomposed as

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

where `Q` is the state, `f` is the full tendency and
`l` is the chosen implicit operator. When `split_explicit_implicit == true`
the assumed decomposition is

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t)]
```

where `f` is now only the nonlinear tendency. For both decompositions the implicit
operator `l` is integrated implicitly whereas the remaining part is integrated
explicitly. Other arguments are the required time step size `dt` and the
optional initial time `t0`. The resulting backward Euler type systems are solved
using the provided `backward_euler_solver`. This time stepping object is
intended to be passed to the `solve!` command.

The constructor builds an additive Runge--Kutta scheme based on the provided
`RKAe`, `RKAi`, `RKB` and `RKC` coefficient arrays.  Additionally `variant`
specifies which of the analytically equivalent but numerically different
formulations of the scheme is used.

The available concrete implementations are:

  - [`ARK1ForwardBackwardEuler`](@ref)
  - [`ARK2ImplicitExplicitMidpoint`](@ref)
  - [`ARK2GiraldoKellyConstantinescu`](@ref)
  - [`ARK548L2SA2KennedyCarpenter`](@ref)
  - [`ARK437L2SA1KennedyCarpenter`](@ref)
  - [`Trap2LockWoodWeller`](@ref)
"""
mutable struct AdditiveRungeKutta{
    T,
    RT,
    AT,
    V,
    VS,
    IST,
    Nstages,
    Nstages_sq,
    Nstagesm1,
} <: AbstractAdditiveRungeKutta
    "time step"
    dt::RT
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "rhs function"
    rhs!::Any
    "rhs linear operator"
    rhs_implicit!::Any
    "a dictionary of backward Euler solvers"
    implicit_solvers::IST
    "Storage for solution during the AdditiveRungeKutta update"
    Qstages::NTuple{Nstagesm1, AT}
    "Storage for RHS during the AdditiveRungeKutta update"
    Rstages::NTuple{Nstages, AT}
    "Storage for the linear solver rhs vector"
    Qhat::AT
    "RK coefficient matrix A for the explicit scheme"
    RKA_explicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient matrix A for the implicit scheme"
    RKA_implicit::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient vector B for the explicit scheme (rhs add in scaling)"
    RKB_explicit::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector B for the implicit scheme (rhs add in scaling)"
    RKB_implicit::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK_explicit coefficient vector C for the explicit scheme (time scaling)"
    RKC_explicit::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK_implicit coefficient vector C for the implicit scheme (time scaling)"
    RKC_implicit::SArray{Tuple{Nstages}, RT, 1, Nstages}
    split_explicit_implicit::Bool
    "Variant of the ARK scheme"
    variant::V
    "Storage dependent on the variant of the ARK scheme"
    variant_storage::VS

    function AdditiveRungeKutta(
        rhs!,
        rhs_implicit!,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q::AT;
        dt = nothing,
        t0 = 0,
        nsubsteps = [],
    ) where {AT <: AbstractArray}

        @assert dt !== nothing

        T = eltype(Q)
        RT = real(T)

        Nstages = length(RKB_explicit)

        Qstages = ntuple(i -> similar(Q), Nstages - 1)
        Rstages = ntuple(i -> similar(Q), Nstages)
        Qhat = similar(Q)

        V = typeof(variant)
        variant_storage = additional_storage(variant, Q, Nstages)
        VS = typeof(variant_storage)

        implicit_solvers = Dict()

        rk_diag = unique(diag(RKA_implicit))
        # Remove all zero entries from `rk_diag`
        # so we build all unique implicit solvers (parameterized by the
        # corresponding RK coefficient)
        filter!(c -> !iszero(c), rk_diag)

        # LowStorageVariant ARK methods assume that both the explicit and
        # implicit B and C vectors are the same. Additionally, the diagonal
        # of the implicit Butcher table A is assumed to have the form:
        # [0, c, ... c ], where c is some non-zero constant.
        if variant isa LowStorageVariant
            @assert RKB_explicit == RKB_implicit
            @assert RKC_explicit == RKC_implicit
            # rk_diag here has been filtered of all non-unique and zero values.
            # So [0, c, ... c ] filters to [c]
            if length(rk_diag) != 1
                error("The implicit Butcher table must have SDIRK form when using LowStorageVariant.")
            end
        end

        if isempty(nsubsteps)
            for rk_coeff in rk_diag
                α = dt * rk_coeff
                besolver! = setup_backward_Euler_solver(
                    backward_euler_solver,
                    Q,
                    α,
                    rhs_implicit!,
                )
                @assert besolver! isa AbstractBackwardEulerSolver
                implicit_solvers[rk_coeff] = besolver!
            end
        else
            for rk_coeff in rk_diag
                α = dt * nsubsteps[i] * rk_coeff
                besolver! = setup_backward_Euler_solver(
                    backward_euler_solver,
                    Q,
                    α,
                    rhs_implicit!,
                )
                @assert besolver! isa AbstractBackwardEulerSolver
                implicit_solvers[rk_coeff] = besolver!
            end
        end

        IST = typeof(implicit_solvers)
        new{T, RT, AT, V, VS, IST, Nstages, Nstages^2, Nstages - 1}(
            RT(dt),
            RT(t0),
            0,
            rhs!,
            rhs_implicit!,
            implicit_solvers,
            Qstages,
            Rstages,
            Qhat,
            RKA_explicit,
            RKA_implicit,
            RKB_explicit,
            RKB_implicit,
            RKC_explicit,
            RKC_implicit,
            split_explicit_implicit,
            variant,
            variant_storage,
        )
    end
end

function AdditiveRungeKutta(
    ark,
    op::TimeScaledRHS{2, RT} where {RT},
    backward_euler_solver,
    Q::AT;
    dt = 0,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = true,
    variant = NaiveVariant(),
) where {AT <: AbstractArray}
    return ark(
        op.rhs![1],
        op.rhs![2],
        backward_euler_solver,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
        split_explicit_implicit = split_explicit_implicit,
        variant = variant,
    )
end

# this will only work for iterative solves
# direct solvers use prefactorization
function updatedt!(ark::AdditiveRungeKutta, dt)
    for (rk_coeff, implicit_solver!) in ark.implicit_solvers
        @assert Δt_is_adjustable(implicit_solver!)
        # New coefficient
        α = dt * rk_coeff
        # Update with new dt and implicit coefficient
        ark.dt = dt
        update_backward_Euler_solver!(implicit_solver!, ark.Qstages[1], α)
    end
end

function dostep!(
    Q,
    ark::AdditiveRungeKutta,
    p,
    time,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    dostep!(Q, ark, ark.variant, p, time, slow_δ, slow_rv_dQ, slow_scaling)
end

function dostep!(
    Q,
    ark::AdditiveRungeKutta,
    p,
    time::Real,
    nsubsteps::Int,
    iStage::Int,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    ark.besolver! = ark.besolvers![iStage]
    for i in 1:nsubsteps
        dostep!(Q, ark, ark.variant, p, time, slow_δ, slow_rv_dQ, slow_scaling)
        time += ark.dt
    end
end

function dostep!(
    Q,
    ark::AdditiveRungeKutta,
    variant::NaiveVariant,
    p,
    time::Real,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    dt = ark.dt

    RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
    RKB_explicit, RKC_explicit = ark.RKB_explicit, ark.RKC_explicit
    RKB_implicit, RKC_implicit = ark.RKB_implicit, ark.RKC_implicit
    rhs!, rhs_implicit! = ark.rhs!, ark.rhs_implicit!
    Qstages, Rstages = (Q, ark.Qstages...), ark.Rstages
    Qhat = ark.Qhat
    split_explicit_implicit = ark.split_explicit_implicit
    Lstages = ark.variant_storage.Lstages

    rv_Q = realview(Q)
    rv_Qstages = realview.(Qstages)
    rv_Lstages = realview.(Lstages)
    rv_Rstages = realview.(Rstages)
    rv_Qhat = realview(Qhat)

    Nstages = length(RKB_explicit)

    groupsize = 256

    # calculate the rhs at first stage to initialize the stage loop
    rhs!(
        Rstages[1],
        Qstages[1],
        p,
        time + RKC_explicit[1] * dt,
        increment = false,
    )

    rhs_implicit!(
        Lstages[1],
        Qstages[1],
        p,
        time + RKC_implicit[1] * dt,
        increment = false,
    )

    # note that it is important that this loop does not modify Q!
    for istage in 2:Nstages
        stagetime_implicit = time + RKC_implicit[istage] * dt
        stagetime_explicit = time + RKC_explicit[istage] * dt

        # this kernel also initializes Qstages[istage] with an initial guess
        # for the linear solver
        event = Event(array_device(Q))
        event = stage_update!(array_device(Q), groupsize)(
            variant,
            rv_Q,
            rv_Qstages,
            rv_Lstages,
            rv_Rstages,
            rv_Qhat,
            RKA_explicit,
            RKA_implicit,
            dt,
            Val(istage),
            Val(split_explicit_implicit),
            slow_δ,
            slow_rv_dQ;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)

        # solves
        # Qs = Qhat + dt * RKA_implicit[istage, istage] * rhs_implicit!(Qs)
        rk_coeff = RKA_implicit[istage, istage]
        if !iszero(rk_coeff)
            α = rk_coeff * dt
            besolver! = ark.implicit_solvers[rk_coeff]
            besolver!(Qstages[istage], Qhat, α, p, stagetime_implicit)
        end

        rhs!(
            Rstages[istage],
            Qstages[istage],
            p,
            stagetime_explicit,
            increment = false,
        )
        rhs_implicit!(
            Lstages[istage],
            Qstages[istage],
            p,
            stagetime_implicit,
            increment = false,
        )
    end

    # compose the final solution
    event = Event(array_device(Q))
    event = solution_update!(array_device(Q), groupsize)(
        variant,
        rv_Q,
        rv_Lstages,
        rv_Rstages,
        RKB_explicit,
        RKB_implicit,
        dt,
        Val(Nstages),
        Val(split_explicit_implicit),
        slow_δ,
        slow_rv_dQ,
        slow_scaling;
        ndrange = length(rv_Q),
        dependencies = (event,),
    )
    wait(array_device(Q), event)
end

function dostep!(
    Q,
    ark::AdditiveRungeKutta,
    variant::LowStorageVariant,
    p,
    time::Real,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    dt = ark.dt

    RKA_explicit, RKA_implicit = ark.RKA_explicit, ark.RKA_implicit
    # LowStorageVariant ARK methods assumes that the implicit
    # Butcher table has an SDIRK form; meaning explicit first step (no
    # implicit solve at the first stage) and all non-zero diaognal
    # coefficients are the same.
    rk_coeff = RKA_implicit[2, 2]
    besolver! = ark.implicit_solvers[rk_coeff]
    # NOTE: Using low-storage variant assumes that the butcher tables
    # for both the explicit and implicit parts have the same B and C
    # vectors
    RKB, RKC = ark.RKB_explicit, ark.RKC_explicit
    rhs!, rhs_implicit! = ark.rhs!, ark.rhs_implicit!
    Qstages, Rstages = (Q, ark.Qstages...), ark.Rstages
    Qhat = ark.Qhat
    split_explicit_implicit = ark.split_explicit_implicit
    Qtt = ark.variant_storage.Qtt

    rv_Q = realview(Q)
    rv_Qstages = realview.(Qstages)
    rv_Rstages = realview.(Rstages)
    rv_Qhat = realview(Qhat)
    rv_Qtt = realview(Qtt)

    Nstages = length(RKB)

    groupsize = 256

    # calculate the rhs at first stage to initialize the stage loop
    rhs!(Rstages[1], Qstages[1], p, time + RKC[1] * dt, increment = false)

    # note that it is important that this loop does not modify Q!
    for istage in 2:Nstages
        stagetime = time + RKC[istage] * dt

        # this kernel also initializes Qtt for the linear solver
        event = Event(array_device(Q))
        event = stage_update!(array_device(Q), groupsize)(
            variant,
            rv_Q,
            rv_Qstages,
            rv_Rstages,
            rv_Qhat,
            rv_Qtt,
            RKA_explicit,
            RKA_implicit,
            dt,
            Val(istage),
            Val(split_explicit_implicit),
            slow_δ,
            slow_rv_dQ;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Q), event)

        # solves
        # Q_tt = Qhat + dt * RKA_implicit[istage, istage] * rhs_implicit!(Q_tt)
        α = dt * RKA_implicit[istage, istage]
        besolver!(Qtt, Qhat, α, p, stagetime)

        # update Qstages
        Qstages[istage] .+= Qtt

        rhs!(Rstages[istage], Qstages[istage], p, stagetime, increment = false)
    end

    if split_explicit_implicit
        for istage in 1:Nstages
            stagetime = time + RKC[istage] * dt
            rhs_implicit!(
                Rstages[istage],
                Qstages[istage],
                p,
                stagetime,
                increment = true,
            )
        end
    end

    # compose the final solution
    event = Event(array_device(Q))
    event = solution_update!(array_device(Q), groupsize)(
        variant,
        rv_Q,
        rv_Rstages,
        RKB,
        dt,
        Val(Nstages),
        slow_δ,
        slow_rv_dQ,
        slow_scaling;
        ndrange = length(rv_Q),
        dependencies = (event,),
    )
    wait(array_device(Q), event)
end

@kernel function stage_update!(
    ::NaiveVariant,
    Q,
    Qstages,
    Lstages,
    Rstages,
    Qhat,
    RKA_explicit,
    RKA_implicit,
    dt,
    ::Val{is},
    ::Val{split_explicit_implicit},
    slow_δ,
    slow_dQ,
) where {is, split_explicit_implicit}
    i = @index(Global, Linear)
    @inbounds begin
        Qhat_i = Q[i]
        Qstages_is_i = Q[i]

        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end

        @unroll for js in 1:(is - 1)
            R_explicit = dt * RKA_explicit[is, js] * Rstages[js][i]
            L_explicit = dt * RKA_explicit[is, js] * Lstages[js][i]
            L_implicit = dt * RKA_implicit[is, js] * Lstages[js][i]
            Qhat_i += (R_explicit + L_implicit)
            Qstages_is_i += R_explicit
            if split_explicit_implicit
                Qstages_is_i += L_explicit
            else
                Qhat_i -= L_explicit
            end
        end
        Qstages[is][i] = Qstages_is_i
        Qhat[i] = Qhat_i
    end
end

@kernel function stage_update!(
    ::LowStorageVariant,
    Q,
    Qstages,
    Rstages,
    Qhat,
    Qtt,
    RKA_explicit,
    RKA_implicit,
    dt,
    ::Val{is},
    ::Val{split_explicit_implicit},
    slow_δ,
    slow_dQ,
) where {is, split_explicit_implicit}
    i = @index(Global, Linear)
    @inbounds begin
        Qhat_i = Q[i]
        Qstages_is_i = -zero(eltype(Q))

        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end

        @unroll for js in 1:(is - 1)
            if split_explicit_implicit
                rkcoeff = RKA_implicit[is, js] / RKA_implicit[is, is]
            else
                rkcoeff =
                    (RKA_implicit[is, js] - RKA_explicit[is, js]) /
                    RKA_implicit[is, is]
            end
            commonterm = rkcoeff * Qstages[js][i]
            Qhat_i += commonterm + dt * RKA_explicit[is, js] * Rstages[js][i]
            Qstages_is_i -= commonterm
        end
        Qstages[is][i] = Qstages_is_i
        Qhat[i] = Qhat_i
        Qtt[i] = Qhat_i
    end
end

@kernel function solution_update!(
    ::NaiveVariant,
    Q,
    Lstages,
    Rstages,
    RKB_explicit,
    RKB_implicit,
    dt,
    ::Val{Nstages},
    ::Val{split_explicit_implicit},
    slow_δ,
    slow_dQ,
    slow_scaling,
) where {Nstages, split_explicit_implicit}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        @unroll for is in 1:Nstages
            Q[i] += RKB_explicit[is] * dt * Rstages[is][i]
            if split_explicit_implicit
                Q[i] += RKB_implicit[is] * dt * Lstages[is][i]
            end
        end
    end
end

@kernel function solution_update!(
    ::LowStorageVariant,
    Q,
    Rstages,
    RKB,
    dt,
    ::Val{Nstages},
    slow_δ,
    slow_dQ,
    slow_scaling,
) where {Nstages}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        @unroll for is in 1:Nstages
            Q[i] += RKB[is] * dt * Rstages[is][i]
        end
    end
end

"""
    ARK1ForwardBackwardEuler(f, l, backward_euler_solver, Q; dt, t0,
                             split_explicit_implicit, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses a first-order-accurate two-stage additive Runge--Kutta scheme
by combining a forward Euler explicit step with a backward Euler implicit
correction.

### References
    @article{Ascher1997,
      title = {Implicit-explicit Runge-Kutta methods for time-dependent
               partial differential equations},
      author = {Uri M. Ascher and Steven J. Ruuth and Raymond J. Spiteri},
      volume = {25},
      number = {2-3},
      pages = {151--167},
      year = {1997},
      journal = {Applied Numerical Mathematics},
      publisher = {Elsevier {BV}}
    }
"""
function ARK1ForwardBackwardEuler(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = LowStorageVariant(),
) where {AT <: AbstractArray}

    @assert dt !== nothing

    T = eltype(Q)
    RT = real(T)

    RKA_explicit = [
        RT(0) RT(0)
        RT(1) RT(0)
    ]
    RKA_implicit = [
        RT(0) RT(0)
        RT(0) RT(1)
    ]

    RKB_explicit = [RT(0), RT(1)]
    RKC_explicit = [RT(0), RT(1)]
    # For this ARK method, both RK methods share the same
    # B and C vectors in the Butcher table
    RKB_implicit = RKB_explicit
    RKC_implicit = RKC_explicit

    Nstages = length(RKB_explicit)

    AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end

"""
    ARK2ImplicitExplicitMidpoint(f, l, backward_euler_solver, Q; dt, t0,
                                 split_explicit_implicit, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses a second-order-accurate two-stage additive Runge--Kutta scheme
by combining the implicit and explicit midpoint methods.

### References
    @article{Ascher1997,
      title = {Implicit-explicit Runge-Kutta methods for time-dependent
               partial differential equations},
      author = {Uri M. Ascher and Steven J. Ruuth and Raymond J. Spiteri},
      volume = {25},
      number = {2-3},
      pages = {151--167},
      year = {1997},
      journal = {Applied Numerical Mathematics},
      publisher = {Elsevier {BV}}
    }
"""
function ARK2ImplicitExplicitMidpoint(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = LowStorageVariant(),
) where {AT <: AbstractArray}

    @assert dt !== nothing

    T = eltype(Q)
    RT = real(T)

    RKA_explicit = [
        RT(0) RT(0)
        RT(1 / 2) RT(0)
    ]
    RKA_implicit = [
        RT(0) RT(0)
        RT(0) RT(1 / 2)
    ]

    RKB_explicit = [RT(0), RT(1)]
    RKC_explicit = [RT(0), RT(1 / 2)]
    # For this ARK method, both RK methods share the same
    # B and C vectors in the Butcher table
    RKB_implicit = RKB_explicit
    RKC_implicit = RKC_explicit

    Nstages = length(RKB_explicit)

    AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end

"""
    ARK2GiraldoKellyConstantinescu(f, l, backward_euler_solver, Q; dt, t0,
                                   split_explicit_implicit, variant, paperversion)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

`paperversion=true` uses the coefficients from the paper, `paperversion=false`
uses coefficients that make the scheme (much) more stable but less accurate

This uses the second-order-accurate 3-stage additive Runge--Kutta scheme of
Giraldo, Kelly and Constantinescu (2013).

### References
 - [Giraldo2013](@cite)
"""
function ARK2GiraldoKellyConstantinescu(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = LowStorageVariant(),
    paperversion = false,
) where {AT <: AbstractArray}

    @assert dt !== nothing

    T = eltype(Q)
    RT = real(T)

    a32 = RT(paperversion ? (3 + 2 * sqrt(2)) / 6 : 1 // 2)
    RKA_explicit = [
        RT(0) RT(0) RT(0)
        RT(2 - sqrt(2)) RT(0) RT(0)
        RT(1 - a32) RT(a32) RT(0)
    ]

    RKA_implicit = [
        RT(0) RT(0) RT(0)
        RT(1 - 1 / sqrt(2)) RT(1 - 1 / sqrt(2)) RT(0)
        RT(1 / (2 * sqrt(2))) RT(1 / (2 * sqrt(2))) RT(1 - 1 / sqrt(2))
    ]

    RKB_explicit =
        [RT(1 / (2 * sqrt(2))), RT(1 / (2 * sqrt(2))), RT(1 - 1 / sqrt(2))]
    RKC_explicit = [RT(0), RT(2 - sqrt(2)), RT(1)]
    # For this ARK method, both RK methods share the same
    # B and C vectors in the Butcher table
    RKB_implicit = RKB_explicit
    RKC_implicit = RKC_explicit

    Nstages = length(RKB_explicit)

    AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end

"""
    Trap2LockWoodWeller(F, L, backward_euler_solver, Q; dt, t0, nsubsteps,
                        split_explicit_implicit, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

The time integrator scheme used is Trap2(2,3,2) with δ_s = 1, δ_f = 0, from
the following reference

### References
    @article{Ascher1997,
      title = {Numerical analyses of Runge–Kutta implicit–explicit schemes
               for horizontally explicit, vertically implicit solutions of
               atmospheric models},
      author = {S.-J. Lock and N. Wood and H. Weller},
      volume = {140},
      number = {682},
      pages = {1654-1669},
      year = {2014},
      journal = {Quarterly Journal of the Royal Meteorological Society},
      publisher = {{RMetS}}
    }
"""
function Trap2LockWoodWeller(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = NaiveVariant(),
    δ_s = 1,
    δ_f = 0,
    α = 0,
) where {AT <: AbstractArray}

    @assert dt !== nothing
    # In this scheme B and C vectors do not coincide,
    # hence we can't use the LowStorageVariant optimization
    @assert variant isa NaiveVariant

    T = eltype(Q)
    RT = real(T)

    #! format: off
    RKA_explicit = [
        RT(0)    RT(0)    RT(0)    RT(0)
        RT(δ_s)  RT(0)    RT(0)    RT(0)
        RT(1/2)  RT(1/2)  RT(0)    RT(0)
        RT(1/2)  RT(0)    RT(1/2)  RT(0)
    ]

    RKA_implicit = [
        RT(0)                  RT(0)                  RT(0)    RT(0)
        RT(δ_f * (1 - α) / 2)  RT(δ_f * (1 + α) / 2)  RT(0)    RT(0)
        RT(1/2)                RT(0)                  RT(1/2)  RT(0)
        RT(1/2)                RT(0)                  RT(0)    RT(1/2)
    ]
    #! format: on

    RKB_explicit = [RT(1 / 2), RT(0), RT(1 / 2), RT(0)]
    RKB_implicit = [RT(1 / 2), RT(0), RT(0), RT(1 / 2)]
    RKC_explicit = [RT(0), RT(δ_s), RT(1), RT(1)]
    RKC_implicit = [RT(0), RT(δ_f), RT(1), RT(1)]

    Nstages = length(RKB_explicit)

    AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end

"""
    ARK548L2SA2KennedyCarpenter(f, l, backward_euler_solver, Q; dt, t0,
                                split_explicit_implicit, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses the fifth-order-accurate 8-stage additive Runge--Kutta scheme of
Kennedy and Carpenter (2013).

### References
 - [Kennedy2019](@cite)
"""
function ARK548L2SA2KennedyCarpenter(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = LowStorageVariant(),
) where {AT <: AbstractArray}

    @assert dt !== nothing

    T = eltype(Q)
    RT = real(T)

    Nstages = 8
    gamma = RT(2 // 9)

    # declared as Arrays for mutability, later these will be converted to static
    # arrays
    RKA_explicit = zeros(RT, Nstages, Nstages)
    RKA_implicit = zeros(RT, Nstages, Nstages)
    RKB_explicit = zeros(RT, Nstages)
    RKC_explicit = zeros(RT, Nstages)

    # the main diagonal
    for is in 2:Nstages
        RKA_implicit[is, is] = gamma
    end

    RKA_implicit[3, 2] = RT(2366667076620 // 8822750406821)
    RKA_implicit[4, 2] = RT(-257962897183 // 4451812247028)
    RKA_implicit[4, 3] = RT(128530224461 // 14379561246022)
    RKA_implicit[5, 2] = RT(-486229321650 // 11227943450093)
    RKA_implicit[5, 3] = RT(-225633144460 // 6633558740617)
    RKA_implicit[5, 4] = RT(1741320951451 // 6824444397158)
    RKA_implicit[6, 2] = RT(621307788657 // 4714163060173)
    RKA_implicit[6, 3] = RT(-125196015625 // 3866852212004)
    RKA_implicit[6, 4] = RT(940440206406 // 7593089888465)
    RKA_implicit[6, 5] = RT(961109811699 // 6734810228204)
    RKA_implicit[7, 2] = RT(2036305566805 // 6583108094622)
    RKA_implicit[7, 3] = RT(-3039402635899 // 4450598839912)
    RKA_implicit[7, 4] = RT(-1829510709469 // 31102090912115)
    RKA_implicit[7, 5] = RT(-286320471013 // 6931253422520)
    RKA_implicit[7, 6] = RT(8651533662697 // 9642993110008)

    RKA_explicit[3, 1] = RT(1 // 9)
    RKA_explicit[3, 2] = RT(1183333538310 // 1827251437969)
    RKA_explicit[4, 1] = RT(895379019517 // 9750411845327)
    RKA_explicit[4, 2] = RT(477606656805 // 13473228687314)
    RKA_explicit[4, 3] = RT(-112564739183 // 9373365219272)
    RKA_explicit[5, 1] = RT(-4458043123994 // 13015289567637)
    RKA_explicit[5, 2] = RT(-2500665203865 // 9342069639922)
    RKA_explicit[5, 3] = RT(983347055801 // 8893519644487)
    RKA_explicit[5, 4] = RT(2185051477207 // 2551468980502)
    RKA_explicit[6, 1] = RT(-167316361917 // 17121522574472)
    RKA_explicit[6, 2] = RT(1605541814917 // 7619724128744)
    RKA_explicit[6, 3] = RT(991021770328 // 13052792161721)
    RKA_explicit[6, 4] = RT(2342280609577 // 11279663441611)
    RKA_explicit[6, 5] = RT(3012424348531 // 12792462456678)
    RKA_explicit[7, 1] = RT(6680998715867 // 14310383562358)
    RKA_explicit[7, 2] = RT(5029118570809 // 3897454228471)
    RKA_explicit[7, 3] = RT(2415062538259 // 6382199904604)
    RKA_explicit[7, 4] = RT(-3924368632305 // 6964820224454)
    RKA_explicit[7, 5] = RT(-4331110370267 // 15021686902756)
    RKA_explicit[7, 6] = RT(-3944303808049 // 11994238218192)
    RKA_explicit[8, 1] = RT(2193717860234 // 3570523412979)
    RKA_explicit[8, 2] = RKA_explicit[8, 1]
    RKA_explicit[8, 3] = RT(5952760925747 // 18750164281544)
    RKA_explicit[8, 4] = RT(-4412967128996 // 6196664114337)
    RKA_explicit[8, 5] = RT(4151782504231 // 36106512998704)
    RKA_explicit[8, 6] = RT(572599549169 // 6265429158920)
    RKA_explicit[8, 7] = RT(-457874356192 // 11306498036315)

    RKB_explicit[2] = 0
    RKB_explicit[3] = RT(3517720773327 // 20256071687669)
    RKB_explicit[4] = RT(4569610470461 // 17934693873752)
    RKB_explicit[5] = RT(2819471173109 // 11655438449929)
    RKB_explicit[6] = RT(3296210113763 // 10722700128969)
    RKB_explicit[7] = RT(-1142099968913 // 5710983926999)

    RKC_explicit[2] = RT(4 // 9)
    RKC_explicit[3] = RT(6456083330201 // 8509243623797)
    RKC_explicit[4] = RT(1632083962415 // 14158861528103)
    RKC_explicit[5] = RT(6365430648612 // 17842476412687)
    RKC_explicit[6] = RT(18 // 25)
    RKC_explicit[7] = RT(191 // 200)

    for is in 2:Nstages
        RKA_implicit[is, 1] = RKA_implicit[is, 2]
    end

    for is in 1:(Nstages - 1)
        RKA_implicit[Nstages, is] = RKB_explicit[is]
    end

    RKB_explicit[1] = RKB_explicit[2]
    RKB_explicit[8] = gamma

    RKA_explicit[2, 1] = RKC_explicit[2]
    RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

    RKC_explicit[1] = 0
    RKC_explicit[Nstages] = 1

    # conversion to static arrays
    RKA_explicit = SMatrix{Nstages, Nstages}(RKA_explicit)
    RKA_implicit = SMatrix{Nstages, Nstages}(RKA_implicit)
    RKB_explicit = SVector{Nstages}(RKB_explicit)
    RKC_explicit = SVector{Nstages}(RKC_explicit)

    # For this ARK method, both RK methods share the same
    # B and C vectors in the Butcher table
    RKB_implicit = RKB_explicit
    RKC_implicit = RKC_explicit

    ark = AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end

"""
    ARK437L2SA1KennedyCarpenter(f, l, backward_euler_solver, Q; dt, t0,
                                split_explicit_implicit, variant)

This function returns an [`AdditiveRungeKutta`](@ref) time stepping object,
see the documentation of [`AdditiveRungeKutta`](@ref) for arguments definitions.
This time stepping object is intended to be passed to the `solve!` command.

This uses the fourth-order-accurate 7-stage additive Runge--Kutta scheme of
Kennedy and Carpenter (2013).

### References
 - [Kennedy2019](@cite)
"""
function ARK437L2SA1KennedyCarpenter(
    F,
    L,
    backward_euler_solver,
    Q::AT;
    dt = nothing,
    t0 = 0,
    nsubsteps = [],
    split_explicit_implicit = false,
    variant = LowStorageVariant(),
) where {AT <: AbstractArray}

    @assert dt !== nothing

    T = eltype(Q)
    RT = real(T)

    Nstages = 7
    gamma = RT(1235 // 10000)

    # declared as Arrays for mutability, later these will be converted to static
    # arrays
    RKA_explicit = zeros(RT, Nstages, Nstages)
    RKA_implicit = zeros(RT, Nstages, Nstages)
    RKB_explicit = zeros(RT, Nstages)
    RKC_explicit = zeros(RT, Nstages)

    # the main diagonal
    for is in 2:Nstages
        RKA_implicit[is, is] = gamma
    end

    RKA_implicit[3, 2] = RT(624185399699 // 4186980696204)
    RKA_implicit[4, 2] = RT(1258591069120 // 10082082980243)
    RKA_implicit[4, 3] = RT(-322722984531 // 8455138723562)
    RKA_implicit[5, 2] = RT(-436103496990 // 5971407786587)
    RKA_implicit[5, 3] = RT(-2689175662187 // 11046760208243)
    RKA_implicit[5, 4] = RT(4431412449334 // 12995360898505)
    RKA_implicit[6, 2] = RT(-2207373168298 // 14430576638973)
    RKA_implicit[6, 3] = RT(242511121179 // 3358618340039)
    RKA_implicit[6, 4] = RT(3145666661981 // 7780404714551)
    RKA_implicit[6, 5] = RT(5882073923981 // 14490790706663)
    RKA_implicit[7, 2] = 0
    RKA_implicit[7, 3] = RT(9164257142617 // 17756377923965)
    RKA_implicit[7, 4] = RT(-10812980402763 // 74029279521829)
    RKA_implicit[7, 5] = RT(1335994250573 // 5691609445217)
    RKA_implicit[7, 6] = RT(2273837961795 // 8368240463276)

    RKA_explicit[3, 1] = RT(247 // 4000)
    RKA_explicit[3, 2] = RT(2694949928731 // 7487940209513)
    RKA_explicit[4, 1] = RT(464650059369 // 8764239774964)
    RKA_explicit[4, 2] = RT(878889893998 // 2444806327765)
    RKA_explicit[4, 3] = RT(-952945855348 // 12294611323341)
    RKA_explicit[5, 1] = RT(476636172619 // 8159180917465)
    RKA_explicit[5, 2] = RT(-1271469283451 // 7793814740893)
    RKA_explicit[5, 3] = RT(-859560642026 // 4356155882851)
    RKA_explicit[5, 4] = RT(1723805262919 // 4571918432560)
    RKA_explicit[6, 1] = RT(6338158500785 // 11769362343261)
    RKA_explicit[6, 2] = RT(-4970555480458 // 10924838743837)
    RKA_explicit[6, 3] = RT(3326578051521 // 2647936831840)
    RKA_explicit[6, 4] = RT(-880713585975 // 1841400956686)
    RKA_explicit[6, 5] = RT(-1428733748635 // 8843423958496)
    RKA_explicit[7, 2] = RT(760814592956 // 3276306540349)
    RKA_explicit[7, 3] = RT(-47223648122716 // 6934462133451)
    RKA_explicit[7, 4] = RT(71187472546993 // 9669769126921)
    RKA_explicit[7, 5] = RT(-13330509492149 // 9695768672337)
    RKA_explicit[7, 6] = RT(11565764226357 // 8513123442827)

    RKB_explicit[2] = 0
    RKB_explicit[3] = RT(9164257142617 // 17756377923965)
    RKB_explicit[4] = RT(-10812980402763 // 74029279521829)
    RKB_explicit[5] = RT(1335994250573 // 5691609445217)
    RKB_explicit[6] = RT(2273837961795 // 8368240463276)
    RKB_explicit[7] = RT(247 // 2000)

    RKC_explicit[2] = RT(247 // 1000)
    RKC_explicit[3] = RT(4276536705230 // 10142255878289)
    RKC_explicit[4] = RT(67 // 200)
    RKC_explicit[5] = RT(3 // 40)
    RKC_explicit[6] = RT(7 // 10)

    for is in 2:Nstages
        RKA_implicit[is, 1] = RKA_implicit[is, 2]
    end

    for is in 1:(Nstages - 1)
        RKA_implicit[Nstages, is] = RKB_explicit[is]
    end

    RKB_explicit[1] = RKB_explicit[2]

    RKA_explicit[2, 1] = RKC_explicit[2]
    RKA_explicit[Nstages, 1] = RKA_explicit[Nstages, 2]

    RKC_explicit[1] = 0
    RKC_explicit[Nstages] = 1

    # conversion to static arrays
    RKA_explicit = SMatrix{Nstages, Nstages}(RKA_explicit)
    RKA_implicit = SMatrix{Nstages, Nstages}(RKA_implicit)
    RKB_explicit = SVector{Nstages}(RKB_explicit)
    RKC_explicit = SVector{Nstages}(RKC_explicit)

    # For this ARK method, both RK methods share the same
    # B and C vectors in the Butcher table
    RKB_implicit = RKB_explicit
    RKC_implicit = RKC_explicit

    ark = AdditiveRungeKutta(
        F,
        L,
        backward_euler_solver,
        RKA_explicit,
        RKA_implicit,
        RKB_explicit,
        RKB_implicit,
        RKC_explicit,
        RKC_implicit,
        split_explicit_implicit,
        variant,
        Q;
        dt = dt,
        t0 = t0,
        nsubsteps = nsubsteps,
    )
end
