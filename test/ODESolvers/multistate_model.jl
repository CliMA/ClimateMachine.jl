using StaticArrays
using CLIMA.VariableTemplates
using CLIMA.DGmethods.NumericalFluxes:
    NumericalFluxNonDiffusive, NumericalFluxDiffusive, NumericalFluxGradient

import CLIMA.DGmethods:
    BalanceLaw,
    vars_state,
    init_state!,
    vars_aux,
    init_aux!,
    LocalGeometry,
    vars_diffusive,
    vars_gradient,
    flux_nondiffusive!,
    flux_diffusive!,
    source!,
    boundary_state!,
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

struct FastODE{T} <: BalanceLaw
    Ω::SMatrix{2, 2, T, 4}
    ω::T
end
struct SlowODE{T} <: BalanceLaw
    Ω::SMatrix{2, 2, T, 4}
    ω::T
end

vars_state(::FastODE, FT) = @vars(Qᶠ::FT)
vars_state(::SlowODE, FT) = @vars(Qˢ::FT)

exactsolution(ω, t) = [sqrt(3 + cos(ω * t)); sqrt(2 + cos(t))]

function init_state!(m::FastODE, Q::Vars, A::Vars, coords, t::Real)
    Q.Qᶠ = exactsolution(m.ω, t)[1]

    return nothing
end

function init_state!(m::SlowODE, Q::Vars, A::Vars, coords, t::Real)
    Q.Qˢ = exactsolution(m.ω, t)[2]

    return nothing
end

vars_aux(::FastODE, FT) = @vars(Qˢ::FT)
vars_aux(::SlowODE, FT) = @vars(Qᶠ::FT)

function init_aux!(m::FastODE, A::Vars, geom::LocalGeometry)
    A.Qˢ = 0

    return nothing
end

function init_aux!(m::SlowODE, A::Vars, geom::LocalGeometry)
    A.Qᶠ = 0

    return nothing
end

vars_diffusive(::FastODE, FT) = @vars()
vars_diffusive(::SlowODE, FT) = @vars()
vars_gradient(::FastODE, FT) = @vars()
vars_gradient(::SlowODE, FT) = @vars()


flux_nondiffusive!(::FastODE, _...) = nothing
flux_nondiffusive!(::SlowODE, _...) = nothing
flux_diffusive!(::FastODE, _...) = nothing
flux_diffusive!(::SlowODE, _...) = nothing

Gᶠ(Qᶠ, ω, t) = (-3 + Qᶠ^2 - cos(ω * t)) / 2Qᶠ
Gˢ(Qˢ, _, t) = (-2 + Qˢ^2 - cos(t)) / 2Qˢ

@inline function source!(m::FastODE, S, Q, D, A, t, direction)
    @inbounds begin
        Qᶠ = Q.Qᶠ
        Qˢ = A.Qˢ
        Ω = m.Ω
        ω = m.ω

        S.Qᶠ += Ω[1, 1] * Gᶠ(Qᶠ, ω, t)
        S.Qᶠ += Ω[1, 2] * Gˢ(Qˢ, ω, t)
        S.Qᶠ -= ω * sin(ω * t) / 2Qᶠ

        return nothing
    end
end

@inline function source!(m::SlowODE, S, Q, D, A, t, direction)
    @inbounds begin
        Qᶠ = A.Qᶠ
        Qˢ = Q.Qˢ
        Ω = m.Ω
        ω = m.ω

        S.Qˢ += Ω[2, 1] * Gᶠ(Qᶠ, ω, t)
        S.Qˢ += Ω[2, 2] * Gˢ(Qˢ, ω, t)
        S.Qˢ -= sin(t) / 2Qˢ

        return nothing
    end
end

function boundary_state!(nf, m::FastODE, _...)
    return nothing
end

function boundary_state!(nf, m::SlowODE, _...)
    return nothing
end

@inline function initialize_fast_state!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
)
    dgSlow.auxstate.Qᶠ .= Qfast.Qᶠ
    dgFast.auxstate.Qˢ .= Qslow.Qˢ

    return nothing
end

@inline function pass_tendency_from_slow_to_fast!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qfast,
    dQslow, # probably not needed here, need Qslow instead
)
    # not sure if this is needed at all
    return nothing
end

@inline function cummulate_fast_solution!(
    fast::FastODE,
    dgFast,
    Qfast,
    fast_time,
    fast_dt,
    total_fast_step,
)
    # not sure if this is needed at all
    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    total_fast_step,
)
    # not sure if this is needed at all
    return nothing
end
