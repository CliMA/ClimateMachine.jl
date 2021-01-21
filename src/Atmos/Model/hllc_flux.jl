
"""
NumericalFluxFirstOrder()
    ::HLLCNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn,
    normal_vector,
    state_prognostic⁻,
    state_auxiliary⁻,
    state_prognostic⁺,
    state_auxiliary⁺,
    t,
    direction,
)

An implementation of the numerical flux based on the HLLC method for
the AtmosModel. For more information on this particular implementation,
see Chapter 10.4 in the provided reference below.

## References

- [Toro2013](@cite)

"""
function numerical_flux_first_order!(
    ::HLLCNumericalFlux,
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
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    param_set = balance_law.param_set

    # Extract the first-order fluxes from the AtmosModel (underlying BalanceLaw)
    # and compute normals on the positive + and negative - sides of the
    # interior facets
    flux⁻ = similar(parent(fluxᵀn), Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    fluxᵀn⁻ = flux⁻' * normal_vector

    flux⁺ = similar(flux⁻)
    fill!(flux⁺, -zero(FT))
    flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn⁺ = flux⁺' * normal_vector

    # Extract relevant fields and thermodynamic variables defined on
    # the positive + and negative - sides of the interior facets
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
    c⁻ = soundspeed_air(ts⁻)

    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = air_pressure(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.ρe
    ts⁺ = recover_thermo_state(
        balance_law,
        balance_law.moisture,
        state_prognostic⁺,
        state_auxiliary⁺,
    )

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    # Wave speeds estimates S⁻ and S⁺
    S⁻ = min(uᵀn⁻ - c⁻, uᵀn⁺ - c⁺)
    S⁺ = max(uᵀn⁻ + c⁻, uᵀn⁺ + c⁺)

    # Compute the middle wave speed S⁰ in the contact/star region
    S⁰ =
        (p⁺ - p⁻ + ρ⁻ * uᵀn⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * uᵀn⁺ * (S⁺ - uᵀn⁺)) /
        (ρ⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * (S⁺ - uᵀn⁺))

    p⁰ =
        (
            p⁺ +
            p⁻ +
            ρ⁻ * (S⁻ - uᵀn⁻) * (S⁰ - uᵀn⁻) +
            ρ⁺ * (S⁺ - uᵀn⁺) * (S⁰ - uᵀn⁺)
        ) / 2

    # Compute p * D = p * (0, n₁, n₂, n₃, S⁰)
    pD = @MVector zeros(FT, num_state_prognostic)
    if balance_law.ref_state isa HydrostaticState &&
       balance_law.ref_state.subtract_off
        # pressure should be continuous but it doesn't hurt to average
        ref_p⁻ = state_auxiliary⁻.ref_state.p
        ref_p⁺ = state_auxiliary⁺.ref_state.p
        ref_p⁰ = (ref_p⁻ + ref_p⁺) / 2

        momentum_p = p⁰ - ref_p⁰
    else
        momentum_p = p⁰
    end

    pD[2] = momentum_p * normal_vector[1]
    pD[3] = momentum_p * normal_vector[2]
    pD[4] = momentum_p * normal_vector[3]
    pD[5] = p⁰ * S⁰

    # Computes both +/- sides of intermediate flux term flux⁰
    flux⁰⁻ =
        (S⁰ * (S⁻ * parent(state_prognostic⁻) - fluxᵀn⁻) + S⁻ * pD) / (S⁻ - S⁰)
    flux⁰⁺ =
        (S⁰ * (S⁺ * parent(state_prognostic⁺) - fluxᵀn⁺) + S⁺ * pD) / (S⁺ - S⁰)

    if 0 <= S⁻
        parent(fluxᵀn) .= fluxᵀn⁻
    elseif S⁻ < 0 <= S⁰
        parent(fluxᵀn) .= flux⁰⁻
    elseif S⁰ < 0 <= S⁺
        parent(fluxᵀn) .= flux⁰⁺
    else # 0 > S⁺
        parent(fluxᵀn) .= fluxᵀn⁺
    end
end
