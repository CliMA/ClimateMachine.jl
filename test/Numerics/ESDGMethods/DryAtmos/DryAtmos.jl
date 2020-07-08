using ClimateMachine.VariableTemplates: Vars, Grad, @vars
import ClimateMachine.BalanceLaws:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_entropy,
    state_to_entropy_variables!,
    entropy_variables_to_state!,
    init_state_auxiliary!,
    state_to_entropy
using StaticArrays: SVector
using LinearAlgebra: dot, I
import ClimateMachine.DGMethods.NumericalFluxes:
    EntropyConservative,
    numerical_volume_conservative_flux_first_order!,
    numerical_volume_fluctuation_flux_first_order!,
    ave,
    logave
using ClimateMachine.Orientations:
    Orientation, FlatOrientation, SphericalOrientation

# Gas constant
const _γ = 7 // 5
const _grav = 981 // 1000

struct DryAtmosModel{D, O} <: BalanceLaw
    orientation::O
end
DryAtmosModel{D}(orientation::O) where {D, O <: Orientation} =
    DryAtmosModel{D, O}(orientation)

"""
    init_state_auxiliary!(
        m::DryAtmosModel,
        aux::Vars,
        geom::LocalGeometry
        )

Initialize geopotential for the `DryAtmosModel`.
"""
function init_state_auxiliary!(
    ::DryAtmosModel{dim, FlatOrientation},
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    grav = FT(_grav)
    @inbounds r = geom.coord[dim]
    state_auxiliary.Φ = grav * r
end
function init_state_auxiliary!(
    ::DryAtmosModel{dim, SphericalOrientation},
    state_auxiliary,
    geom,
) where {dim}
    FT = eltype(state_auxiliary)
    grav = FT(_grav)
    r = norm(geom.coord)
    state_auxiliary.Φ = grav * r
end

"""
    pressure(ρ, ρu, ρe, Φ)

Compute the pressure given density `ρ`, momentum `ρu`, total energy `ρe`, and
gravitational potential `Φ`.
"""
function pressure(ρ, ρu, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(_γ)
    (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

"""
    totalenergy(ρ, ρu, p, Φ)

Compute the total energy given density `ρ`, momentum `ρu`, pressure `p`, and
gravitational potential `Φ`.
"""
function totalenergy(ρ, ρu, p, Φ)
    FT = eltype(ρ)
    γ = FT(_γ)
    return p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
end

"""
    soundspeed(ρ, p)

Compute the speed of sound from the density `ρ` and pressure `p`.
"""
function soundspeed(ρ, p)
    FT = eltype(ρ)
    γ = FT(_γ)
    sqrt(γ * p / ρ)
end

"""
    vars_state_conservative(::DryAtmosModel, FT)

The state variables for the `DryAtmosModel` are density `ρ`, momentum `ρu`,
and total energy `ρe`
"""
function vars_state_conservative(::DryAtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end

"""
    vars_state_auxiliary(::DryAtmosModel, FT)

The auxiliary variables for the `DryAtmosModel` is gravitational potential
`Φ`
"""
function vars_state_auxiliary(::DryAtmosModel, FT)
    @vars begin
        Φ::FT
    end
end

"""
    vars_state_entropy(::DryAtmosModel, FT)

The entropy variables for the `DryAtmosModel` correspond to the state
variables density `ρ`, momentum `ρu`, and total energy `ρe` as well as the
auxiliary variable gravitational potential `Φ`
"""
function vars_state_entropy(::DryAtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        Φ::FT
    end
end

"""
    state_to_entropy_variables!(
        ::DryAtmosModel,
        entropy::Vars,
        state::Vars,
        aux::Vars,
    )

See [`BalanceLaws.state_to_entropy_variables!`](@ref)
"""
function state_to_entropy_variables!(
    ::DryAtmosModel,
    entropy::Vars,
    state::Vars,
    aux::Vars,
)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ

    FT = eltype(state)
    γ = FT(_γ)

    p = pressure(ρ, ρu, ρe, Φ)
    s = log(p / ρ^γ)
    b = ρ / 2p
    u = ρu / ρ

    entropy.ρ = (γ - s) / (γ - 1) - (dot(u, u) - 2Φ) * b
    entropy.ρu = 2b * u
    entropy.ρe = -2b
    entropy.Φ = 2ρ * b
end

"""
    entropy_variables_to_state!(
        ::DryAtmosModel,
        state::Vars,
        aux::Vars,
        entropy::Vars,
    )

See [`BalanceLaws.entropy_variables_to_state!`](@ref)
"""
function entropy_variables_to_state!(
    ::DryAtmosModel,
    state::Vars,
    aux::Vars,
    entropy::Vars,
)
    FT = eltype(state)
    β = entropy
    γ = FT(_γ)

    b = -β.ρe / 2
    ρ = β.Φ / (2b)
    ρu = ρ * β.ρu / (2b)

    p = ρ / (2b)
    s = log(p / ρ^γ)
    Φ = dot(ρu, ρu) / (2 * ρ^2) - ((γ - s) / (γ - 1) - β.ρ) / (2b)

    ρe = p / (γ - 1) + dot(ρu, ρu) / (2ρ) + ρ * Φ

    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
    aux.Φ = Φ
end

function state_to_entropy(::DryAtmosModel, state::Vars, aux::Vars)
    FT = eltype(state)
    ρ, ρu, ρe, Φ = state.ρ, state.ρu, state.ρe, aux.Φ
    p = pressure(ρ, ρu, ρe, Φ)
    γ = FT(_γ)
    s = log(p * ρ^γ)
    η = -ρ * s
    return η
end


function numerical_volume_conservative_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    F::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(F)
    ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
    ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    u_1 = ρu_1 / ρ_1
    u_2 = ρu_2 / ρ_2
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
    b_1 = ρ_1 / 2p_1
    b_2 = ρ_2 / 2p_2

    ρ_avg = ave(ρ_1, ρ_2)
    u_avg = ave(u_1, u_2)
    b_avg = ave(b_1, b_2)
    Φ_avg = ave(Φ_1, Φ_2)

    usq_avg = ave(dot(u_1, u_1), dot(u_2, u_2))

    ρ_log = logave(ρ_1, ρ_2)
    b_log = logave(b_1, b_2)
    α = b_avg * ρ_log / 2b_1

    γ = FT(_γ)

    Fρ = u_avg * ρ_log
    Fρu = u_avg * Fρ' + ρ_avg / 2b_avg * I
    Fρe = (1 / (2 * (γ - 1) * b_log) - usq_avg / 2 + Φ_avg) * Fρ + Fρu * u_avg

    F.ρ += Fρ
    F.ρu += Fρu
    F.ρe += Fρe
end

function numerical_volume_fluctuation_flux_first_order!(
    ::EntropyConservative,
    ::DryAtmosModel,
    D::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(D)
    ρ_1, ρu_1, ρe_1 = state_1.ρ, state_1.ρu, state_1.ρe
    ρ_2, ρu_2, ρe_2 = state_2.ρ, state_2.ρu, state_2.ρe
    Φ_1, Φ_2 = aux_1.Φ, aux_2.Φ
    p_1 = pressure(ρ_1, ρu_1, ρe_1, Φ_1)
    p_2 = pressure(ρ_2, ρu_2, ρe_2, Φ_2)
    b_1 = ρ_1 / 2p_1
    b_2 = ρ_2 / 2p_2

    ρ_log = logave(ρ_1, ρ_2)
    b_avg = ave(b_1, b_2)
    α = b_avg * ρ_log / 2b_1

    D.ρu -= α * (Φ_1 - Φ_2) * I
end
