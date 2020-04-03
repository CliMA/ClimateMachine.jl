using StaticArrays
using CLIMA.VariableTemplates

import CLIMA.DGmethods:
    BalanceLaw,
    vars_state,
    init_state!,
    vars_aux,
    init_aux!,
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

struct MultiODE{T} <: BalanceLaw
    Ω::SMatrix{2, 2, T, 4}
    ω::T
end

vars_state(::MultiODE, FT) = @vars(Qᶠ::FT, Qˢ::FT)

exactsolution(ω, t) = [sqrt(3 + cos(ω * t)); sqrt(2 + cos(t))]

function init_state!(m::MultiODE, Q::Vars, A::Vars, coords, t::Real)
    Q.Qᶠ = exactsolution(m.ω, t)[1]
    Q.Qˢ = exactsolution(m.ω, t)[2]

    return nothing
end

vars_aux(::MultiODE, FT) = @vars()
init_aux!(::MultiODE, _...) = nothing

vars_diffusive(::MultiODE, FT) = @vars()
vars_gradient(::MultiODE, FT) = @vars()

flux_nondiffusive!(::MultiODE, _...) = nothing
flux_diffusive!(::MultiODE, _...) = nothing

Gᶠ(Qᶠ, ω, t) = (-3 + Qᶠ^2 - cos(ω * t)) / 2Qᶠ
Gˢ(Qˢ, _, t) = (-2 + Qˢ^2 - cos(t)) / 2Qˢ

@inline function source!(m::MultiODE, S, Q, D, A, t, direction)
    @inbounds begin
        Qᶠ = Q.Qᶠ
        Qˢ = Q.Qˢ
        Ω = m.Ω
        ω = m.ω

        S.Qᶠ += Ω[1, 1] * Gᶠ(Qᶠ, ω, t)
        S.Qᶠ += Ω[1, 2] * Gˢ(Qˢ, ω, t)
        S.Qᶠ -= ω * sin(ω * t) / 2Qᶠ

        S.Qˢ += Ω[2, 1] * Gᶠ(Qᶠ, ω, t)
        S.Qˢ += Ω[2, 2] * Gˢ(Qˢ, ω, t)
        S.Qˢ -= sin(t) / 2Qˢ

        return nothing
    end
end

boundary_state!(nf, m::MultiODE, _...) = nothing

struct NullODE <: BalanceLaw end

vars_state(::NullODE, FT) = @vars()
init_state!(m::NullODE, _...) = nothing
vars_aux(::NullODE, FT) = @vars()
init_aux!(m::NullODE, _...) = nothing
vars_diffusive(::NullODE, FT) = @vars()
vars_gradient(::NullODE, FT) = @vars()
flux_nondiffusive!(::NullODE, _...) = nothing
flux_diffusive!(::NullODE, _...) = nothing
source!(::NullODE, _...) = nothing
boundary_state!(nf, ::NullODE, _...) = nothing

initialize_fast_state!(::Union{NullODE, MultiODE}, _...) = nothing
pass_tendency_from_slow_to_fast!(::Union{NullODE, MultiODE}, _...) = nothing
cummulate_fast_solution!(::Union{NullODE, MultiODE}, _...) = nothing
reconcile_from_fast_to_slow!(::Union{NullODE, MultiODE}, _...) = nothing
