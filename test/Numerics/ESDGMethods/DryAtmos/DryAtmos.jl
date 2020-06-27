using ClimateMachine.VariableTemplates: Vars, Grad, @vars
import ClimateMachine.BalanceLaws:
    BalanceLaw,
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_entropy,
    state_to_entropy_variables!,
    entropy_variables_to_state!
using StaticArrays: SVector
using LinearAlgebra: dot

struct DryAtmosModel <: BalanceLaw end

# Gas constant
const _γ = 7 // 5

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
