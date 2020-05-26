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
    # Q - how to pass 
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
    # Parameters
    cv_d::FT = cv_d(param_set)
    cv_v::FT = cp_v(param_set)
    cv_l::FT = cp_l(param_set)
    cv_i::FT = cp_i(param_set)
    T0::FT = T_0(param_set)
    e_int_i0::FT = e_int_i0(param_set)
    ϵ_v = 1. / molmass_ratio(param_set)
    g = FT(grav(param_set))
    R_d = FT(R_d(param_set))

    ρinv = 1/gm.ρ
    L = Vector(undef, 3)
    a_L = model.a_L(obukhov_length)
    b_L = model.b_L(obukhov_length)

    # precompute
    en_area = 1-sum([up[i].ρa for i in 1:N])*ρinv
    w_env = (gm.ρu[3]-sum([up[i].ρau[3] for i in 1:N]))*ρinv
    en_e_int = (gm.ρe_int-up[i].ρae_int)/(gm.ρ*up_area)
    en_q_tot = (gm.ρq_tot-up[i].ρaq_tot)/(gm.ρ*up_area)
    TKE_Shear = en_d.∇u[1].^2 + en_d.∇u[2].^2 + en_d.∇u[3].^2
    ∂e_int∂z = en_d.e_int
    ∂q_tot∂z = en_d.q_tot
    
 
    # Thermodynamic local variables for mixing length
    ts = PhaseEquil(param_set ,en_e_int, gm.ρ, en_q_tot)
    Π = exner(ts)
    lv = latent_heat_vapor(ts)
    cp_m_ = cp_m(ts)
    tke = sqrt(max(en.ρatke, FT(0))*ρinv)
    θ_ρ = virtual_pottemp(ts)
    R_m = gas_constant_air(ts)
    T = air_temperature(ts)
    q_vap = vapor_specific_humidity(ts)
    q_liq = PhasePartition(ts).liq
    q_ice = PhasePartition(ts).ice

    # compute L1 - static stability
    buoyancy_freq = g*en_d.∇θ_ρ/θ_ρ
    if buoyancy_freq>0
      L[1] = sqrt(m.c_w*tke)/buoyancy_freq
    else
      L[1] = 1e-6
    end

    # compute L2 - law of the wall
    if obukhov_length < 0.0 #unstable case
      L[2] = (m.κ * z/(sqrt(tke)/m.ustar/m.ustar)* m.c_k) * min(
         (1 - 100 * z/obukhov_length)^0.2, 1/m.κ))
    else # neutral or stable cases
      L[2] = m.κ * z/(sqrt(max(q[:tke, k_1, en], FT(0))/m.ustar/m.ustar)*m.c_k)
    end

    # I think this is an alrenative way for the same computation
    ξ = z/obukhov_length
    κ_star = m.ustar/sqrt(tke)
    L[2] = m.κ*z/(m.c_k*κ_star*ϕ_m(ξ, a_L, b_L))

    # compute L3 - entrainment detrainment sources
    
    # buoyancy gradients via chain-role
    ∂b∂ρ = - g/gm.ρ
    #                  <-------- ∂ρ∂T -------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_dry    = - R_d*gm_a.p_0/(R_m*T*T)/((1-en_q_tot)*cv_d+q_vap *cv_v)
    #                  <-------- ∂ρ∂T --------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_cloudy = - (R_d*gm_a.p_0/(R_m*T*T)/((1-en_q_tot)*cv_d+q_vap *cv_v+q_liq*cv_l+ q_ice*cv_i)
                       + gm_a.p_0/(R_m*R_m*T)*ϵ_v*R_d/(cv_v*(T-T0)+e_int_i0) )  
    #                    <----- ∂ρ∂Rm ------->*<------- ∂Rm∂e_int ---------->
    
    ∂ρ∂e_int = (en_a.cld_frac * ∂ρ∂e_int_cloudy + (1-en_a.cld_frac) * ∂ρ∂e_int_dry)
    ∂ρ∂q_tot = R_d*gm_a.p_0/(R_m*R_m*T)
    # apply chain-role
    ∂b∂z_e_int = ∂b∂ρ * ∂ρ∂e_int * ∂e_int∂z
    ∂b∂z_q_tot = ∂b∂ρ * ∂ρ∂q_tot * ∂q_tot∂z

    Grad_Ri = min(∂b∂z_e_int/max(TKE_Shear, eps(FT)) + ∂b∂z_q_tot/max(TKE_Shear, eps(FT)) , 0.25)
    if unstable(obukhov_length)
      Pr_z = m.Pr_n
    else
      Pr_z = m.Pr_n*(2*Grad_Ri/
                        (1+(53/13)*Grad_Ri -sqrt( (1+(53/130)*Grad_Ri)^2 - 4*Grad_Ri ) ) )
    end
    # Production/destruction terms
    a = m.c_m*(TKE_Shear - ∂b∂z_e_int/Pr_z - ∂b∂z_q_tot/Pr_z)* sqrt(tke)
    # Dissipation term
    b = 0.0

    for i in 1:N
      a_up = up[i].ρa/gm.ρ
      w_up = up[i].ρau[3]/up[i].ρa
      b += a_up*w_up*δ/en_area*((w_up-w_env)*(w_up-w_env)/2-tke) - a_up*w_up*(w_up-w_env)*εt*w_env/en_area
    end

    c_neg = m.c_m*tke*sqrt(tke)
    if abs(a) > eps(FT) && 4*a*c_neg > - b^2
              l_entdet = max( -b/2.0/a + sqrt(b^2 + 4*a*c_neg)/2/a, 0)
    elseif abs(a) < eps(FT) && abs(b) > eps(FT)
              l_entdet = c_neg/b
    else
      l_entdet = 0.0
    end
    L[3] = l_entdet

    # lamb_smooth_minimum
    lower_bound = 0.1
    upper_bound = 1.5
    leng = 3
    xmin = minimum(L)
    lambda0 = max(xmin*lower_bound/real(LambertW.lambertw(2.0/MathConstants.e)), upper_bound)

    i = 1
    num = 0
    den = 0
    while(tuple(i)<leng)
      num += l[i]*exp(-(l[i]-xmin)/lambda0)
      den += exp(-(l[i]-xmin)/lambda0)
      i += 1
    end
    l = num/den

    return l
end;