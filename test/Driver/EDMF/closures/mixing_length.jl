#### Mixing length model kernels

function mixing_length(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    δ::FT,
    εt::FT,
    ) where {FT, N}

    # Q - do I need to define L as a vector ?
    # need to code the functions: obukhov_length, ustar, ϕ_m, lamb_smooth_minimum

    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft
    en_d = state.edmf.environment.diffusive

    z = gm_a.z
    _grav = FT(grav(param_set))
    bflux = compute_blux(_grav, ϵ_v)
    # ustar = compute_ustar(??) ustar should be comepute from a function 
    obukhov_length = compute_MO_len(m.κ, m.ustar, bflux)

    ρinv = 1/gm.ρ
    fill!(m.L, 0)

    # precompute
    en_area  = 1-sum([up[i].ρa for i in 1:N])*ρinv
    w_env    = (gm.ρu[3]-sum([up[i].ρau[3] for i in 1:N]))*ρinv
    en_e_int = (gm.ρe_int-sum([up[i].ρae_int for i in 1:N]))*ρinv
    en_q_tot = (gm.ρq_tot-sum([up[i].ρaq_tot for i in 1:N]))*ρinv
    ∂e_int∂z = en_d.e_int
    ∂q_tot∂z = en_d.q_tot

    TKE_Shear = en_d.∇u[1].^2 + en_d.∇u[2].^2 + en_d.∇u[3].^2

    # Thermodynamic local variables for mixing length
    ts = PhaseEquil(param_set ,en_e_int, gm.ρ, en_q_tot)
    Π = exner(ts)
    lv = latent_heat_vapor(ts)
    cp_m_ = cp_m(ts)
    tke = sqrt(en.ρatke, FT(0))*ρinv/en_area
    θ_ρ = virtual_pottemp(ts)

    # compute L1 - static stability
    buoyancy_freq = g*en_d.∇θ_ρ/θ_ρ
    if buoyancy_freq>FT(0)
      m.L[1] = sqrt(m.c_w*tke)/buoyancy_freq
    else
      m.L[1] = FT(1e-6)
    end

    # compute L2 - law of the wall
    if obukhov_length < FT(0) #unstable case
      m.L[2] = (m.κ * z/(sqrt(tke)/m.ustar/m.ustar)* m.c_k) * min(
         (FT(1) - FT(100) * z/obukhov_length)^FT(0.2), 1/m.κ))
    else # neutral or stable cases
      m.L[2] = m.κ * z/(sqrt(max(q[:tke, k_1, en], FT(0))/m.ustar/m.ustar)*m.c_k)
    end

    # compute L3 - entrainment detrainment sources

    # buoyancy gradients via chain-role
    ∂b∂z_e_int, ∂b∂z_q_tot = compute_buoyancy_gradients(ss, m,source,state, diffusive, aux, t, direction)
    Grad_Ri = gradient_Richardson_number(∂b∂z_e_int, TKE_Shear, ∂b∂z_q_tot, FT(0.25)) # this parameter should be exposed in the model 
    Pr_z = turbulent_Prandtl_number(m.Pr_n, Grad_Ri, obukhov_length, FT(53),FT(13),FT(130)) # these parameters should be exposed in the model 

    # Production/destruction terms
    a = m.c_m*(TKE_Shear - ∂b∂z_e_int/Pr_z - ∂b∂z_q_tot/Pr_z)* sqrt(tke)
    # Dissipation term
    b = FT(0)

    for i in 1:N
      a_up = up[i].ρa/gm.ρ
      w_up = up[i].ρau[3]/up[i].ρa
      b += a_up*w_up*δ/en_area*((w_up-w_env)*(w_up-w_env)/2-tke) - a_up*w_up*(w_up-w_env)*εt*w_env/en_area
    end

    c_neg = m.c_m*tke*sqrt(tke)
    if abs(a) > eps(FT) && 4*a*c_neg > - b^2
              l_entdet = max( -b/FT(2)/a + sqrt(b^2 + 4*a*c_neg)/2/a, FT(0))
    elseif abs(a) < eps(FT) && abs(b) > eps(FT)
              l_entdet = c_neg/b
    else
      l_entdet = FT(0)
    end
    m.L[3] = l_entdet

    lower_bound = FT(0.1)
    upper_bound = FT(1.5)

    l = lamb_smooth_minimum(m.L,lower_bound, upper_bound)
    return l
end;