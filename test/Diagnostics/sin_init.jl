function init_sin_test!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    FT = eltype(state)
    param_set = parameter_set(bl)

    z = FT(z)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    # These constants are those used by Stevens et al. (2005)
    qref = FT(9.0e-3)
    q_pt_sfc = PhasePartition(qref)
    Rm_sfc = FT(gas_constant_air(param_set, q_pt_sfc))
    T_sfc = FT(292.5)
    P_sfc = _MSLP

    # Specify moisture profiles
    q_liq = FT(0)
    q_ice = FT(0)
    zb = FT(600)         # initial cloud bottom
    zi = FT(840)         # initial cloud top
    dz_cloud = zi - zb
    q_liq_peak = FT(0.00045)     # cloud mixing ratio at z_i

    if z > zb && z <= zi
        q_liq = (z - zb) * q_liq_peak / dz_cloud
    end

    if z <= zi
        θ_liq = FT(289.0)
        q_tot = qref
    else
        θ_liq = FT(297.5) + (z - zi)^(FT(1 / 3))
        q_tot = FT(1.5e-3)
    end

    w = FT(10 + 0.5 * sin(2 * π * ((x / 1500) + (y / 1500))))
    u = (5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))
    v = FT(5 + 2 * sin(2 * π * ((x / 1500) + (y / 1500))))

    # Pressure
    H = Rm_sfc * T_sfc / _grav
    p = P_sfc * exp(-z / H)

    # Density, Temperature
    ts = PhaseEquil_pθq(param_set, p, θ_liq, q_tot)
    #ρ = air_density(ts)
    ρ = one(FT)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = _grav * z
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.energy.ρe = E
    state.moisture.ρq_tot = ρ * q_tot
end
