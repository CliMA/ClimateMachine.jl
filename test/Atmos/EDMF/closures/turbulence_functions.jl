#### Turbulence model kernels

"""
    thermo_variables(ts::ThermodynamicState)

A NamedTuple of thermodynamic variables,
computed from the given thermodynamic state.
"""
function thermo_variables(ts::ThermodynamicState)
    return (
        θ_dry = dry_pottemp(ts),
        θ_liq = liquid_ice_pottemp(ts),
        q_tot = total_specific_humidity(ts),
        T = air_temperature(ts),
        R_m = gas_constant_air(ts),
        q_vap = vapor_specific_humidity(ts),
        q_liq = liquid_specific_humidity(ts),
        q_ice = ice_specific_humidity(ts),
    )
end

"""
    compute_buoyancy_gradients(
        m::AtmosModel{FT},
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        ts
    ) where {FT}
Returns the environmental buoyancy gradient following Tan et al. (JAMES, 2018)
and the effective environmental static stability following
Lopez-Gomez et al. (JAMES, 2020), given:
 - `m`, an `AtmosModel`
 - `state`, state variables
 - `diffusive`, additional variables
 - `aux`, auxiliary variables
 - `t`, the time
"""
function compute_buoyancy_gradients(
    m::AtmosModel{FT},
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ts,
) where {FT}
    # Alias convention:
    gm = state
    en_dif = diffusive.turbconv.environment
    N_up = n_updrafts(m.turbconv)

    _grav::FT = grav(m.param_set)
    _R_d::FT = R_d(m.param_set)
    _R_v::FT = R_v(m.param_set)
    ε_v::FT = 1 / molmass_ratio(m.param_set)
    p = air_pressure(ts.gm)

    q_tot_en = total_specific_humidity(ts.en)
    θ_liq_en = liquid_ice_pottemp(ts.en)
    lv = latent_heat_vapor(ts.en)
    T = air_temperature(ts.en)
    Π = exner(ts.en)
    q_liq = liquid_specific_humidity(ts.en)
    _cp_m = cp_m(ts.en)
    θ_virt = virtual_pottemp(ts.en)

    (ts_dry, ts_cloudy, cloud_frac) =
        compute_subdomain_statistics(m, state, aux, t)
    cloudy = thermo_variables(ts_cloudy)
    dry = thermo_variables(ts_dry)

    prefactor = _grav * (_R_d * gm.ρ / p * Π)

    ∂b∂θl_dry = prefactor * (1 + (ε_v - 1) * dry.q_tot)
    ∂b∂qt_dry = prefactor * dry.θ_liq * (ε_v - 1)

    if cloud_frac > FT(0)
        num =
            prefactor *
            (1 + ε_v * (1 + lv / _R_v / cloudy.T) * cloudy.q_vap - cloudy.q_tot)
        den = 1 + lv * lv / _cp_m / _R_v / cloudy.T / cloudy.T * cloudy.q_vap
        ∂b∂θl_cloudy = num / den
        ∂b∂qt_cloudy =
            (lv / _cp_m / cloudy.T * ∂b∂θl_cloudy - prefactor) * cloudy.θ_dry
    else
        ∂b∂θl_cloudy = FT(0)
        ∂b∂qt_cloudy = FT(0)
    end

    ∂b∂θl = (cloud_frac * ∂b∂θl_cloudy + (1 - cloud_frac) * ∂b∂θl_dry)
    ∂b∂qt = (cloud_frac * ∂b∂qt_cloudy + (1 - cloud_frac) * ∂b∂qt_dry)

    # Partial buoyancy gradients
    ∂b∂z_θl = en_dif.∇θ_liq[3] * ∂b∂θl
    ∂b∂z_qt = en_dif.∇q_tot[3] * ∂b∂qt
    ∂b∂z = ∂b∂z_θl + ∂b∂z_qt

    # Computation of buoyancy frequency based on θ_lv
    ∂θvl∂θ_liq = 1 + (ε_v - 1) * q_tot_en
    ∂θvl∂qt = (ε_v - 1) * θ_liq_en
    # apply chain-rule
    ∂θvl∂z = ∂θvl∂θ_liq * en_dif.∇θ_liq[3] + ∂θvl∂qt * en_dif.∇q_tot[3]

    ∂θv∂θvl = exp(lv * q_liq / _cp_m / T)
    λ_stb = cloud_frac

    Nˢ_eff =
        _grav / θ_virt *
        ((1 - λ_stb) * en_dif.∇θv[3] + λ_stb * ∂θvl∂z * ∂θv∂θvl)
    return ∂b∂z, Nˢ_eff
end;

"""
    ∇Richardson_number(
        ∂b∂z::FT,
        Shear²::FT,
        minval::FT,
        Ri_c::FT,
    ) where {FT}

Returns the gradient Richardson number, given:
 - `∂b∂z`, the vertical buoyancy gradient
 - `Shear²`, the squared vertical gradient of horizontal velocity
 - `maxval`, maximum value of the output, typically the critical Ri number
"""
function ∇Richardson_number(
    ∂b∂z::FT,
    Shear²::FT,
    minval::FT,
    Ri_c::FT,
) where {FT}
    return min(∂b∂z / max(Shear², minval), Ri_c)
end;

"""
    turbulent_Prandtl_number(
        Pr_n::FT,
        Grad_Ri::FT,
        ω_pr::FT
    ) where {FT}

Returns the turbulent Prandtl number, given:
 - `Pr_n`, the turbulent Prandtl number under neutral conditions
 - `Grad_Ri`, the gradient Richardson number
"""
function turbulent_Prandtl_number(Pr_n::FT, Grad_Ri::FT, ω_pr::FT) where {FT}
    denom = 1 + ω_pr * Grad_Ri - sqrt((1 + ω_pr * Grad_Ri)^2 - 4 * Grad_Ri)
    return Pr_n * 2 * Grad_Ri / denom
end;
