#### Mixing length model kernels
"""
    mixing_length(
        m::AtmosModel{FT},
        ml::MixingLengthModel,
        args,
        Δ::Tuple,
        Et::Tuple,
        ts_gm,
        ts_en,
        env,
    ) where {FT}

Returns the mixing length used in the diffusive turbulence closure, given:
 - `m`, an `AtmosModel`
 - `ml`, a `MixingLengthModel`
 - `args`, the top-level arguments
 - `Δ`, the detrainment rate
 - `Et`, the turbulent entrainment rate
 - `ts_gm`, grid-mean thermodynamic states
 - `ts_en`, environment thermodynamic states
 - `env`, NamedTuple of environment variables
"""
function mixing_length(
    m::AtmosModel{FT},
    ml::MixingLengthModel,
    args,
    Δ::Tuple,
    Et::Tuple,
    ts_gm,
    ts_en,
    env,
) where {FT}
    @unpack state, aux, diffusive, t = args
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

    # buoyancy related functions
    # compute obukhov_length and ustar from SurfaceFlux.jl here
    ustar = m.turbconv.surface.ustar
    obukhov_length = m.turbconv.surface.obukhov_length

    ∂b∂z, Nˢ_eff = compute_buoyancy_gradients(m, args, ts_gm, ts_en)
    Grad_Ri = ∇Richardson_number(∂b∂z, Shear², 1 / ml.max_length, ml.Ri_c)
    Pr_t = turbulent_Prandtl_number(ml.Pr_n, Grad_Ri, ml.ω_pr)

    # compute L1
    Nˢ_fact = (sign(Nˢ_eff - eps(FT)) + 1) / 2
    coeff = min(ml.c_b * sqrt(tke_en) / Nˢ_eff, ml.max_length)
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

    L_W = ml.κ * z / (sqrt(m.turbconv.surface.κ_star²) * ml.c_m)
    if obukhov_length < -eps(FT)
        L_W *= min((FT(1) - ml.a2 * z / obukhov_length)^ml.a1, 1 / ml.κ)
    end

    # compute L3 - entrainment detrainment sources
    # Production/destruction terms
    a = ml.c_m * (Shear² - ∂b∂z / Pr_t) * sqrt(tke_en)
    # Dissipation term
    b = FT(0)
    a_up = vuntuple(i -> up[i].ρa * ρinv, N_up)
    w_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρaw / up[i].ρa)
    end
    b = sum(
        ntuple(N_up) do i
            Δ[i] / gm.ρ / env.a *
            ((w_up[i] - env.w) * (w_up[i] - env.w) / 2 - tke_en) -
            (w_up[i] - env.w) * Et[i] / gm.ρ * env.w / env.a
        end,
    )

    c_neg = ml.c_d * tke_en * sqrt(tke_en)
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
    return l_mix, ∂b∂z, Pr_t
end;
