using ClimateMachine.NumericalFluxes
using ClimateMachine.Thermodynamics

import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

Base.@kwdef struct RoesanovFlux{S,T} <: NumericalFluxFirstOrder
    ω_roe::S = 1.0
    ω_rusanov::T = 1.0
end

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    model::ModelSetup,
    fluxᵀn::Vars{S},
    n⁻::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        model,
        fluxᵀn,
        n⁻,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )
    eos = model.physics.eos

    FT = eltype(fluxᵀn)

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n⁻

    # in general thermodynamics
    p⁻ = calc_pressure(eos, state⁻)
    c⁻ = calc_sound_speed(eos, state⁻)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n⁻

    # in general thermodynamics
    p⁺ = calc_pressure(eos, state⁺)
    c⁺ = calc_sound_speed(eos, state⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n⁻

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n⁻

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    
    fluxᵀn.ρ -= (w1 + w2 + w3) * 0.5
    fluxᵀn.ρu -=
        (
            w1 * (u - c * n⁻) +
            w2 * (u + c * n⁻) +
            w3 * u +
            w4 * (Δu - Δuₙ * n⁻)
        ) * 0.5
    fluxᵀn.ρθ -= ((w1 + w2) * θ + w5) * 0.5

    return nothing
end

function numerical_flux_first_order!(
    ::RoeNumericalFlux,
    balance_law::DryAtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)
    # param_set = parameter_set(balance_law)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    # Φ = gravitational_potential(balance_law, state_auxiliary⁻)
    Φ = state_auxiliary⁻.Φ

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe

    # ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)
    ts⁻ = PhaseDry(param_set, internal_energy(ρ⁻,ρe⁻,ρu⁻,Φ), ρ⁻) 

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = air_pressure(ts⁻)
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    Φ⁺ = state_auxiliary⁺.Φ

    # TODO: state_auxiliary⁺ is not up-to-date
    # with state_prognostic⁺ on the boundaries
    # ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)
    ts⁺ = PhaseDry(param_set, internal_energy(ρ⁺,ρe⁺,ρu⁺,Φ⁺), ρ⁺) 

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuᵀn = Δu' * normal_vector

    w1 = abs(ũᵀn - c̃) * (Δp - ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w2 = abs(ũᵀn + c̃) * (Δp + ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w3 = abs(ũᵀn) * (Δρ - Δp / c̃^2)
    w4 = abs(ũᵀn) * ρ̃

    fluxᵀn.ρ -= (w1 + w2 + w3) / 2
    fluxᵀn.ρu -=
        (
            w1 * (ũ - c̃ * normal_vector) +
            w2 * (ũ + c̃ * normal_vector) +
            w3 * ũ +
            w4 * (Δu - Δuᵀn * normal_vector)
        ) / 2
    fluxᵀn.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - _T_0 * _cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2

end

function numerical_flux_first_order!(
    ϕ::RoesanovFlux,
    model::ModelSetup,
    fluxᵀn::Vars{S},
    n⁻::SVector,
    state⁻::Vars{S},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    aux⁺::Vars{A},
    t,
    direction,
) where {S, A}
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        model,
        fluxᵀn,
        n⁻,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )
    eos = model.physics.eos

    FT = eltype(fluxᵀn)

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻
    uₙ⁻ = u⁻' * n⁻

    # in general thermodynamics
    p⁻ = calc_pressure(eos, state⁻)
    c⁻ = calc_sound_speed(eos, state⁻)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺
    uₙ⁺ = u⁺' * n⁻

    # in general thermodynamics
    p⁺ = calc_pressure(eos, state⁺)
    c⁺ = calc_sound_speed(eos, state⁺)

    # construct roe averges
    ρ = sqrt(ρ⁻ * ρ⁺)
    u = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    θ = roe_average(ρ⁻, ρ⁺, θ⁻, θ⁺)
    c = roe_average(ρ⁻, ρ⁺, c⁻, c⁺)

    # construct normal velocity
    uₙ = u' * n⁻

    # differences
    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δρθ = ρθ⁺ - ρθ⁻
    Δuₙ = Δu' * n⁻

    # constructed values
    c⁻² = 1 / c^2
    w1 = abs(uₙ - c) * (Δp - ρ * c * Δuₙ) * 0.5 * c⁻²
    w2 = abs(uₙ + c) * (Δp + ρ * c * Δuₙ) * 0.5 * c⁻²
    w3 = abs(uₙ) * (Δρ - Δp * c⁻²)
    w4 = abs(uₙ) * ρ
    w5 = abs(uₙ) * (Δρθ - θ * Δp * c⁻²)

    # fluxes!!!
    ω1 = ϕ.ω_roe
    fluxᵀn.ρ -= (w1 + w2 + w3) * 0.5 * ω1
    fluxᵀn.ρu -=
        (
            w1 * (u - c * n⁻) +
            w2 * (u + c * n⁻) +
            w3 * u +
            w4 * (Δu - Δuₙ * n⁻)
        ) * 0.5 * ω1
    fluxᵀn.ρθ -= ((w1 + w2) * θ + w5) * 0.5 * ω1
    
    ω2 = ϕ.ω_rusanov
    max_wavespeed = max(c⁻, c⁺)
    Δρu = ρu⁺ - ρu⁻
    fluxᵀn.ρu -=
        (
            max_wavespeed * Δρu
        ) * 0.5 * ω2


    return nothing
end
