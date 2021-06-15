using ClimateMachine.NumericalFluxes

import ClimateMachine.NumericalFluxes: numerical_flux_first_order!

struct RefanovFlux <: NumericalFluxFirstOrder end 

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
    parameters = model.physics.parameters 

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻

    # in general thermodynamics
    p⁻ = calc_pressure(eos, state⁻, aux⁻, parameters)
    c⁻ = calc_sound_speed(eos, state⁻, aux⁻, parameters)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺

    # in general thermodynamics
    p⁺ = calc_pressure(eos, state⁺, aux⁺, parameters)
    c⁺ = calc_sound_speed(eos, state⁺, aux⁺, parameters)

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
    eos = balance_law.physics.eos
    parameters = balance_law.physics.parameters

    cv_d = parameters.cv_d
    T_0  = parameters.T_0

    Φ = state_auxiliary⁻.Φ #Φ⁻ and Φ⁺ have the same value
    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    u⁻ = ρu⁻ / ρ⁻

    p⁻ = calc_pressure(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    c⁻ = calc_sound_speed(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    h⁻ = calc_total_specific_enthalpy(eos, state_prognostic⁻, state_auxiliary⁻, parameters)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    u⁺ = ρu⁺ / ρ⁺

    p⁺ = calc_pressure(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    c⁺ = calc_sound_speed(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    h⁺ = calc_total_specific_enthalpy(eos, state_prognostic⁺, state_auxiliary⁺, parameters)

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
            w3 * (ũ' * ũ / 2 + Φ - T_0 * cv_d) +
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

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρθ⁻ = state⁻.ρθ

    # constructed states
    u⁻ = ρu⁻ / ρ⁻
    θ⁻ = ρθ⁻ / ρ⁻

    # in general thermodynamics
    p⁻ = calc_pressure(eos, state⁻, aux⁻, parameters)
    c⁻ = calc_sound_speed(eos, state⁻, aux⁻, parameters)

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρθ⁺ = state⁺.ρθ

    # constructed states
    u⁺ = ρu⁺ / ρ⁺
    θ⁺ = ρθ⁺ / ρ⁺

    # in general thermodynamics
    p⁺ = calc_pressure(eos, state⁺, aux⁺, parameters)
    c⁺ = calc_sound_speed(eos, state⁺, aux⁺, parameters)

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

function numerical_flux_first_order!(
    ::LMARSNumericalFlux,
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
    FT = eltype(fluxᵀn)
    eos = balance_law.physics.eos
    parameters = balance_law.physics.parameters

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector

    p⁻ = calc_pressure(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    # if the reference state is removed in the momentum equations (meaning p-p_ref is used for pressure gradient force) then we should remove the reference pressure
    #     p⁻ -= state_auxiliary⁻.ref_state.p
    # end
    c⁻ = calc_sound_speed(eos, state_prognostic⁻, state_auxiliary⁻, parameters)
    h⁻ = calc_total_specific_enthalpy(eos, state_prognostic⁻, state_auxiliary⁻, parameters)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector

    p⁺ = calc_pressure(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    # if the reference state is removed in the momentum equations (meaning p-p_ref is used for pressure gradient force) then we should remove the reference pressure
    #     p⁺ -= state_auxiliary⁺.ref_state.p
    # end
    # c⁺ = calc_sound_speed(eos, state_prognostic⁺, state_auxiliary⁺, parameters)
    h⁺ = calc_total_specific_enthalpy(eos, state_prognostic⁺, state_auxiliary⁺, parameters)

    # Eqn (49), (50), β the tuning parameter
    β = FT(1)
    u_half = 1 / 2 * (uᵀn⁺ + uᵀn⁻) - β * 1 / (ρ⁻ + ρ⁺) / c⁻ * (p⁺ - p⁻)
    p_half = 1 / 2 * (p⁺ + p⁻) - β * ((ρ⁻ + ρ⁺) * c⁻) / 4 * (uᵀn⁺ - uᵀn⁻)

    # Eqn (46), (47)
    ρ_b = u_half > FT(0) ? ρ⁻ : ρ⁺
    ρu_b = u_half > FT(0) ? ρu⁻ : ρu⁺
    ρh_b = u_half > FT(0) ? ρ⁻ * h⁻ : ρ⁺ * h⁺

    # Update fluxes Eqn (18)
    fluxᵀn.ρ = ρ_b * u_half
    fluxᵀn.ρu = ρu_b * u_half .+ p_half * normal_vector
    fluxᵀn.ρe = ρh_b * u_half

    # Update fluxes for moisture
    ρq⁻ = state_prognostic⁻.ρq
    q⁻ = ρq⁻ / ρ⁻
    ρq⁺ = state_prognostic⁺.ρq
    q⁺ = ρq⁺ / ρ⁺
    ρq_b = u_half > FT(0) ? ρq⁻ : ρq⁺
    fluxᵀn.ρq = ρq_b * u_half

    # if !(tracer_model(balance_law) isa NoTracers)
    #     ρχ⁻ = state_prognostic⁻.tracers.ρχ
    #     χ⁻ = ρχ⁻ / ρ⁻
    #     ρχ⁺ = state_prognostic⁺.tracers.ρχ
    #     χ⁺ = ρχ⁺ / ρ⁺
    #     ρχ_b = u_half > FT(0) ? ρχ⁻ : ρχ⁺
    #     fluxᵀn.tracers.ρχ = ρχ_b * u_half
    # end
end

numerical_flux_first_order!(::Nothing, ::DryAtmosModel, _...) = nothing
numerical_flux_second_order!(::Nothing, ::DryAtmosModel, _...) = nothing
numerical_boundary_flux_second_order!(::Nothing, a, ::DryAtmosModel, _...) = nothing

function wavespeed(
    model::DryAtmosModel,
    n⁻,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    eos = model.physics.eos
    parameters = model.physics.parameters
    ρ = state.ρ
    ρu = state.ρu

    u = ρu / ρ
    u_norm = abs(dot(n⁻, u))
    return u_norm + calc_sound_speed(eos, state, aux, parameters)
end

function numerical_flux_first_order!(
    ::RefanovFlux,
    model::DryAtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
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
        normal_vector,
        state⁻,
        aux⁻,
        state⁺,
        aux⁺,
        t,
        direction,
    )

    eos = model.physics.eos
    parameters = model.physics.parameters
    
    c⁻ = calc_ref_sound_speed(eos, state⁻, aux⁻, parameters)
    c⁺ = calc_ref_sound_speed(eos, state⁺, aux⁺, parameters)
    c = max(c⁻, c⁺)

    # - states
    ρ⁻ = state⁻.ρ
    ρu⁻ = state⁻.ρu
    ρe⁻ = state⁻.ρe
    ρq⁻ = state⁻.ρq

    # + states
    ρ⁺ = state⁺.ρ
    ρu⁺ = state⁺.ρu
    ρe⁺ = state⁺.ρe
    ρq⁺ = state⁺.ρq

    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
    Δρq = ρq⁺ - ρq⁻

    fluxᵀn.ρ  -= c * Δρ  * 0.5
    fluxᵀn.ρu -= c * Δρu * 0.5
    fluxᵀn.ρe -= c * Δρe * 0.5
    #fluxᵀn.ρq -= c * Δρq * 0.5
end
