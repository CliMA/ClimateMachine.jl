#### Turbulence model kernels

function compute_buoyancy_gradients(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    # think how to call subdomain statistics here to get cloudy and dry values of T if you nee them
    # buoyancy gradients via chain-role
    # Alias convention:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft
    en_d = diffusive.edmf.environment
    gm_d = diffusive
    gm_a = aux

    _cv_d::FT     = cv_d(param_set)
    _cv_v::FT     = cv_v(param_set)
    _cv_l::FT     = cv_l(param_set)
    _cv_i::FT     = cv_i(param_set)
    _T_0::FT      = T_0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    _grav::FT     = grav(param_set)
    _R_d::FT      = R_d(param_set)
    ε_v::FT       = 1 / molmass_ratio(param_set)

    cld_frac ,cloudy_q_tot ,cloudy_T ,cloudy_R_m ,cloudy_q_vap ,cloudy_q_liq ,cloudy_q_ice ,dry_q_tot ,dry_T ,dry_R_m ,dry_q_vap ,dry_q_liq ,dry_q_ice = compute_subdomain_statistics!(ss, state, aux ,t, ss.edmf.micro_phys.statistical_model)
    ∂b∂ρ = - _grav/gm.ρ

    dry_∂e_int∂z = en_d.∇e_int
    dry_∂q_tot∂z = en_d.∇q_tot
    cloudy_∂e_int∂z = en_d.∇e_int
    cloudy_∂q_tot∂z = en_d.∇q_tot
    
    ρ_i = gm_a.p0/(dry_T*dry_R_m) 
    ∂b∂z_dry = - ∂b∂ρ*ρ_i*( 1/( (1-dry_q_tot)*_cv_d*dry_T + dry_q_vap *_cv_v * dry_T) * dry_∂e_int∂z 
                        + (_R_d/dry_R_m)*(ε_v-1)*dry_∂q_tot∂z - gm_d.∇p0/gm_a.p0)

    ρ_i = gm_a.p0/(cloudy_T*cloudy_R_m) 
    ∂b∂z_cloudy = - ∂b∂ρ*ρ_i*(1/( (1-cloudy_q_tot)*_cv_d+cloudy_q_vap*_cv_v + cloudy_q_liq*_cv_l + cloudy_q_ice*_cv_i)/cloudy_T*cloudy_∂e_int∂z  
                            +(_R_d/dry_R_m) * (1/(_cv_v*(cloudy_T-_T_0)+_e_int_i0)*cloudy_∂e_int∂z + (ε_v-1)*cloudy_∂q_tot∂z)
                            - gm_d.∇p0/gm_a.p0)
    # ρ_i = gm_a.p0/(dry.T*dry.R_m) 
    # ∂b∂z_dry = - ∂b∂ρ*ρ_i*( 1/( (1-dry.q_tot)*_cv_d*dry.T + dry.q_vap *_cv_v * dry.T) * dry.∂e_int∂z 
    #                     + (_R_d/dry.R_m)*(ε_v-1)*dry.∂q_tot∂z - gm_d.∇p0/gm_a.p0)

    # ρ_i = gm_a.p0/(cloudy.T*cloudy.R_m) 
    # ∂b∂z_cloudy = - ∂b∂ρ*ρ_i*(1/( (1-cloudy.q_tot)*_cv_d+cloudy.q_vap*_cv_v + cloudy.q_liq*_cv_l + cloudy.q_ice*_cv_i)/cloudy.T*cloudy.∂e_int∂z  
    #                         +(_R_d/dry.R_m) * (1/(_cv_v*(cloudy.T-_T0)+_e_int_i0)*cloudy.∂e_int∂z + (ε_v-1)*cloudy.∂q_tot∂z)
    #                         - gm_d.∇p0/gm_a.p0)
    # combine cloudy and dry
    ∂b∂z = (cld_frac*∂b∂z_cloudy + (1-cld_frac)*∂b∂z_dry)
    
    # keeping the old derivation commented for now 
    # #                  <-------- ∂ρ∂T -------->*<----- ∂T∂e_int ---------->
    # ∂ρ∂e_int_dry    = - _R_d*gm_a.p_0/(dry.R_m*dry.T*dry.T)/((1-dry.q_tot)*_cv_d+dry.q_vap *_cv_v)
    # #                  <-------- ∂ρ∂T --------->*<----- ∂T∂e_int ---------->
    # ∂ρ∂e_int_cloudy = - (_R_d*gm_a.p_0/(cloudy.R_m*cloudy.T*cloudy.T)/((1-cloudy.q_tot)*_cv_d+cloudy.q_vap *_cv_v+cloudy.q_liq*_cv_l+ cloudy.q_ice*_cv_i)
    #                    + gm_a.p_0/(cloudy.R_m*cloudy.R_m*cloudy.T)*ε_v*_R_d/(_cv_v*(cloudy.T-_T0)+_e_int_i0) )
    # #                    <----- ∂ρ∂Rm ------->*<------- ∂Rm∂e_int ---------->

    # ∂ρ∂e_int = (cld_frac * ∂ρ∂e_int_cloudy + (1-cld_frac) * ∂ρ∂e_int_dry)
    # ∂ρ∂q_tot = _R_d*gm_a.p_0/(R_m*R_m*T)
    # # apply chain-role
    # ∂b∂z_e_int = ∂b∂ρ * ∂ρ∂e_int * ∂e_int∂z
    # ∂b∂z_q_tot = ∂b∂ρ * ∂ρ∂q_tot * ∂q_tot∂z

    # Computation of buoyancy frequeacy based on θ_lv
    ρinv = 1/gm.ρ
    en_e_int = (gm.ρe_int-sum([up[j].ρae_int for j in 1:N]))*ρinv
    en_q_tot = (gm.ρq_tot-sum([up[j].ρaq_tot for j in 1:N]))*ρinv
    ts = PhaseEquil(ss.param_set ,en_e_int, gm.ρ, en_q_tot)
    q = PhasePartition(ts)
    _cp_m = cp_m(ss.param_set, q)
    lv = latent_heat_vapor(ts)
    T = air_temperature(ts)
    Π = exner(ts)
    ql = PhasePartition(ts).liq
    θv =  virtual_pottemp(ts)
    Tv =  θv/Π # check if its not *
    θvl =  θv*exp(-(lv*ql)/(_cp_m*T))

    ∂θv∂e_int  = 1/(((1-cloudy_q_tot)*_cv_d+cloudy_q_vap *_cv_v
              + cloudy_q_liq*_cv_l+ cloudy_q_ice*_cv_i))* θv/T*(1-lv*ql/_cp_m*T)
    ∂θvl∂e_int = 1/(((1-cloudy_q_tot)*_cv_d+cloudy_q_vap *_cv_v
               + cloudy_q_liq*_cv_l+ cloudy_q_ice*_cv_i))* θvl/T*(1-lv*ql/_cp_m*T)
    ∂θv∂qt     = -θv/Tv*(ε_v-1)/_e_int_i0
    ∂θvl∂qt    = -θvl/Tv*(ε_v-1)/_e_int_i0 
    # apply chain-role
    ∂θv∂z  = ∂θv∂e_int*dry_∂e_int∂z  + ∂θv∂qt*dry_∂q_tot∂z
    @show(∂θv∂z,∂θv∂e_int,dry_∂e_int∂z , ∂θv∂qt,dry_∂q_tot∂z)
    ∂θvl∂z = ∂θvl∂e_int*dry_∂e_int∂z + ∂θvl∂qt*dry_∂q_tot∂z

    ∂θv∂vl = exp((lv*ql)/(_cp_m*T))
    λ_stb = cld_frac
    
    Nˢ_eff = _grav/θv*( (1-λ_stb)*∂θv∂z + λ_stb*∂θvl∂z*∂θv∂vl)

    return ∂b∂z, Nˢ_eff
end;

function gradient_Richardson_number(∂b∂z, Shear, minval)
    #Grad_Ri = min(∂b∂z/max(Shear, eps(FT)), minval)  - YAIR problems here 
    Grad_Ri = FT(0.25)
    return Grad_Ri
end;

function turbulent_Prandtl_number(Pr_n, Grad_Ri, obukhov_length)
    Pr_z = Pr_n*(2*Grad_Ri/(1+(FT(53)/FT(13))*Grad_Ri -sqrt( (1+(FT(53)/FT(130))*Grad_Ri)^2 - 4*Grad_Ri)))
    # overwrite 
    # if unstable(obukhov_length)
    #   Pr_z = Pr_n
    # else
    #   Pr_z = Pr_n*(2*Grad_Ri/(1+(FT(53)/FT(13))*Grad_Ri -sqrt( (1+(FT(53)/FT(130))*Grad_Ri)^2 - 4*Grad_Ri)))
    # end
    return Pr_z
end;

function compute_windspeed(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    ) where {FT, N}
    windspeed_min = FT(0.01) # does this needs to be exposed ?
  return max(hypot(gm.u, gm.v), windspeed_min)
end;


