#### Mixing length model kernels

"""
    mixing_length(
        m::AtmosModel{FT},
        ml::MixingLengthModel,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        δ::Tuple,
        εt::Tuple,
        ts,
        env,
    ) where {FT}

Returns the mixing length used in the diffusive turbulence closure, given:
 - `m`, an `AtmosModel`
 - `ml`, a `MixingLengthModel`
 - `state`, state variables
 - `diffusive`, additional variables
 - `aux`, auxiliary variables
 - `t`, the time
 - `δ`, the detrainment rate
 - `εt`, the turbulent entrainment rate
 - `ts`, NamedTuple of thermodynamic states
 - `env`, NamedTuple of environment variables
"""
function mixing_length(
    m::AtmosModel{FT},
    ml::MixingLengthModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    δ::Tuple,
    εt::Tuple,
    ts,
    env,
) where {FT}

    # TODO: use functions: obukhov_length, ustar, ϕ_m

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    gm_aux = aux
    N_up = n_updrafts(m.turbconv)

    z = altitude(m, aux)
    _grav::FT = grav(m.param_set)
    ρinv = 1 / gm.ρ

    Shear² = diffusive.turbconv.S²
    tke_en = max(en.ρatke, 0) * ρinv / env.a

    ustar = m.turbconv.surface.ustar
    obukhov_length = m.turbconv.surface.obukhov_length

    # buoyancy related functions
    ∂b∂z, Nˢ_eff = compute_buoyancy_gradients(m, state, diffusive, aux, t, ts)
    Grad_Ri = ∇Richardson_number(∂b∂z, Shear², 1 / ml.max_length, ml.Ri_c)
    Pr_z = turbulent_Prandtl_number(ml.Pr_n, Grad_Ri, ml.ω_pr)

    # compute L1
    Nˢ_fact = (sign(Nˢ_eff - eps(FT)) + 1) / 2
    coeff = min(sqrt(ml.c_b * tke_en) / Nˢ_eff, ml.max_length)
    L_Nˢ = coeff * Nˢ_fact + ml.max_length * (FT(1) - Nˢ_fact)

    # compute L2 - law of the wall
    surf_vals = subdomain_surface_values(
        m.turbconv.surface,
        m.turbconv,
        m,
        gm,
        gm_aux,
        m.turbconv.surface.zLL,
    )
    tke_surf = surf_vals.tke
    L_W = ml.κ * z / (sqrt(tke_surf) * ml.c_m / ustar / ustar)
    stab_fac = -(sign(obukhov_length) - 1) / 2
    L_W *= stab_fac * min((FT(1) - ml.a2 * z / obukhov_length)^ml.a1, 1 / ml.κ)

    # compute L3 - entrainment detrainment sources
    # Production/destruction terms
    a = ml.c_m * (Shear² - ∂b∂z / Pr_z) * sqrt(tke_en)
    # Dissipation term
    b = FT(0)
    a_up = vuntuple(i -> up[i].ρa * ρinv, N_up)
    w_up = vuntuple(i -> up[i].ρaw / up[i].ρa, N_up)
    b = sum(
        ntuple(N_up) do i
            a_up[i] * w_up[i] * δ[i] / env.a *
            ((w_up[i] - env.w) * (w_up[i] - env.w) / 2 - tke_en) -
            a_up[i] * w_up[i] * (w_up[i] - env.w) * εt[i] * env.w / env.a
        end,
    )

    c_neg = ml.c_d * tke_en * sqrt(abs(tke_en))
    if abs(a) > ml.random_minval && 4 * a * c_neg > -b^2
        l_entdet =
            max(-b / FT(2) / a + sqrt(b^2 + 4 * a * c_neg) / 2 / a, FT(0))
    elseif abs(a) < eps(FT) && abs(b) > eps(FT)
        l_entdet = c_neg / b
    else
        l_entdet = FT(0)
    end
    L_tke = l_entdet

    if L_Nˢ < eps(FT) || L_Nˢ > ml.max_length
        L_Nˢ = ml.max_length
    end
    if L_W < eps(FT) || L_W > ml.max_length
        L_W = ml.max_length
    end
    if L_tke < eps(FT) || L_tke > ml.max_length
        L_tke = ml.max_length
    end

    l_mix =
        lamb_smooth_minimum(SVector(L_Nˢ, L_W, L_tke), ml.smin_ub, ml.smin_rm)
    return l_mix
end;
