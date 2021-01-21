roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_penalty!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert balance_law.moisture isa DryModel

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ts⁻ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁻,
        state_auxiliary⁻,
    )

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = air_pressure(ts⁻)
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe

    # TODO: state_auxiliary⁺ is not up-to-date
    # with state_prognostic⁺ on the boundaries
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

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

    if !(balance_law.tracers isa NoTracers)
        ρχ⁻ = state_prognostic⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻

        ρχ⁺ = state_prognostic⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺

        χ̃ = roe_average(ρ⁻, ρ⁺, χ⁻, χ⁺)
        Δρχ = ρχ⁺ - ρχ⁻

        wt = abs(ũᵀn) * (Δρχ - χ̃ * Δp / c̃^2)

        fluxᵀn.tracers.ρχ -= ((w1 + w2) * χ̃ + wt) / 2
    end
end

function numerical_flux_penalty!(
    numerical_flux::RoeNumericalFluxMoist,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    balance_law.moisture isa EquilMoist ||
        error("Must use a EquilMoist model for RoeNumericalFluxMoist")

    FT = eltype(fluxᵀn)
    param_set = balance_law.param_set
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    γ::FT = cp_d(param_set) / cv_d(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.ρe
    ρq_tot⁻ = state_prognostic⁻.moisture.ρq_tot

    u⁻ = ρu⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    qt⁻ = ρq_tot⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    ρq_tot⁺ = state_prognostic⁺.moisture.ρq_tot

    u⁺ = ρu⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    qt⁺ = ρq_tot⁺ / ρ⁺
    c⁺ = soundspeed_air(ts⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    e_tot = roe_average(ρ⁻, ρ⁺, e⁻, e⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    qt = roe_average(ρ⁻, ρ⁺, qt⁻, qt⁺)
    ρ = sqrt(ρ⁻ * ρ⁺)
    e_int⁻ = internal_energy(ts⁻)
    e_int⁺ = internal_energy(ts⁺)
    e_int = roe_average(ρ⁻, ρ⁺, e_int⁻, e_int⁺)
    ts = PhaseEquil(
        param_set,
        e_int,
        ρ,
        qt,
        balance_law.moisture.maxiter,
        balance_law.moisture.tolerance,
    )
    c̃ = sqrt((γ - 1) * (h̃ - (ũ[1]^2 + ũ[2]^2 + ũ[3]^2) / 2))
    (R_m, _cp_m, _cv_m, gamma) = gas_constants(ts)
    # chosen by fair dice roll
    # guaranteed to be random
    ω = FT(π) / 3
    δ = FT(π) / 5
    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
    # tangent space basis
    τ1 = random_unit_vector × normal_vector
    τ2 = τ1 × normal_vector
    ũᵀn⁻ = u⁻' * normal_vector
    ũᵀn⁺ = u⁺' * normal_vector
    ũᵀn = ũ' * normal_vector
    ũc̃⁻ = ũ + c̃ * normal_vector
    ũc̃⁺ = ũ - c̃ * normal_vector
    e_kin_pot = h̃ - _e_int_v0 * qt - _cp_m * c̃^2 / R_m
    if (numerical_flux.LM == true)
        Mach⁺ = sqrt(u⁺' * u⁺) / c⁺
        Mach⁻ = sqrt(u⁻' * u⁻) / c⁻
        Mach = (Mach⁺ + Mach⁻) / 2
        c̃_LM = c̃ * min(Mach * sqrt(4 + (1 - Mach^2)^2) / (1 + Mach^2), 1)
    else
        c̃_LM = c̃
    end
    #Standard Roe
    Λ = SDiagonal(
        abs(ũᵀn - c̃_LM),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn + c̃_LM),
        abs(ũᵀn),
    )
    #Harten Hyman
    if (numerical_flux.HH == true)
        Λ = SDiagonal(
            max(
                abs(ũᵀn - c̃_LM),
                max(
                    0,
                    ũᵀn - c̃_LM - (u⁻' * normal_vector - c⁻),
                    u⁺' * normal_vector - c⁺ - (ũᵀn - c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn + c̃_LM),
                max(
                    0,
                    ũᵀn + c̃_LM - (u⁻' * normal_vector + c⁻),
                    u⁺' * normal_vector + c⁺ - (ũᵀn + c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
        )
    end
    if (numerical_flux.LV == true)
        #Pseudo LeVeque Fix
        δ_L_1 = max(0, ũᵀn - ũᵀn⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - (ũᵀn⁻ - c⁻))
        δ_L_3 = max(0, ũᵀn + c̃_LM - (ũᵀn⁻ + c⁻))
        δ_R_1 = max(0, ũᵀn⁺ - ũᵀn)
        δ_R_2 = max(0, ũᵀn⁺ - c⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, ũᵀn⁺ + c⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃ < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃_LM) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃ > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end
    if (numerical_flux.LVPP == true)
        #PosPreserving with LeVeque
        b_L = min(ũᵀn - c̃_LM, ũᵀn⁻ - c⁻)
        b_R = max(ũᵀn + c̃_LM, ũᵀn⁺ + c⁺)
        b⁻ = min(0, b_L)
        b⁺ = max(0, b_R)
        δ_L_1 = max(0, ũᵀn - b⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - b⁻)
        δ_L_3 = max(0, ũᵀn + c̃_LM - b⁻)
        δ_R_1 = max(0, b⁺ - ũᵀn)
        δ_R_2 = max(0, b⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, b⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃_LM < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃_LM > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end

    M = hcat(
        SVector(1, ũc̃⁺[1], ũc̃⁺[2], ũc̃⁺[3], h̃ - c̃ * ũᵀn, qt),
        SVector(0, τ1[1], τ1[2], τ1[3], τ1' * ũ, 0),
        SVector(0, τ2[1], τ2[2], τ2[3], τ2' * ũ, 0),
        SVector(1, ũ[1], ũ[2], ũ[3], ũ' * ũ / 2 + Φ - _T_0 * _cv_m, 0),
        SVector(1, ũc̃⁻[1], ũc̃⁻[2], ũc̃⁻[3], h̃ + c̃ * ũᵀn, qt),
        SVector(0, 0, 0, 0, _e_int_v0, 1),
    )
    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
    Δρq_tot = ρq_tot⁺ - ρq_tot⁻
    Δstate = SVector(Δρ, Δρu[1], Δρu[2], Δρu[3], Δρe, Δρq_tot)
    parent(fluxᵀn) .-= M * Λ * (M \ Δstate) / 2
end
