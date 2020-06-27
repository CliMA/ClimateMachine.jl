import ClimateMachine.ESDGMethods:
    numerical_volume_fluctuation!,
    logave,
    ave,
    state_to_entropy_variables,
    entropy_to_state_variables
using ClimateMachine: BalanceLaw
using ClimateMachine.VariableTemplates: Vars, Grad, @vars
import ClimateMachine.BalanceLaws:
    vars_state_auxiliary,
    vars_state_conservative,
    number_state_conservative,
    number_state_auxiliary
using StaticArrays: SVector
using LinearAlgebra: dot, I

struct DryAtmosphereModel <: BalanceLaw end

function vars_state_conservative(m::DryAtmosphereModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end

function vars_state_auxiliary(m::DryAtmosphereModel, FT)
    @vars begin
        Φ::FT
    end
end

function vars_state_entropy(m::DryAtmosphereModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        Φ::FT
    end
end

const _γ = 7 // 5
function pressure(ρ, ρu, ρe, Φ)
    FT = eltype(ρ)
    γ = FT(_γ)
    (γ - 1) * (ρe - dot(ρu, ρu) / 2ρ - ρ * Φ)
end

function totalenergy(ρ, ρu, p, Φ)
    FT = eltype(ρ)
    γ = FT(_γ)
    return p / (γ - 1) + dot(ρu, ρu) / 2ρ + ρ * Φ
end

function soundspeed(ρ, p)
    FT = eltype(ρ)
    γ = FT(_γ)
    sqrt(γ * p / ρ)
end

function state_to_entropy_variables(
    ::DryAtmosphereModel,
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

function entropy_to_state_variables(
    ::DryAtmosphereModel,
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
    Φ = ρu^2 / (2 * ρ^2) - ((γ - s) / (γ - 1) - β.ρ) / (2b)

    ρe = p / (γ - 1) + ρu^2 / (2ρ) + ρ * Φ

    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe
    aux.Φ = Φ
end

function numerical_volume_fluctuation!(
    ::DryAtmosphereModel,
    H::Grad,
    state_1::Vars,
    aux_1::Vars,
    state_2::Vars,
    aux_2::Vars,
)
    FT = eltype(H)
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

    H.ρ = Fρ
    H.ρu = Fρu - α * (Φ_1 - Φ_2) * I
    H.ρe = Fρe
end
