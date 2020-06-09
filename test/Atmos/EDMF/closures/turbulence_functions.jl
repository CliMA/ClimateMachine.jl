#### Turbulence model kernels

function compute_buoyancy_gradients(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
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
    _cp_v::FT     = cp_v(param_set)
    _cp_l::FT     = cp_l(param_set)
    _cp_i::FT     = cp_i(param_set)
    _T_0::FT      = T_0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    _grav::FT     = grav(param_set)
    _R_d::FT      = R_d(param_set)
    ε_v::FT       = 1 / molmass_ratio(param_set)

    cloudy, dry, cld_frac = compute_subdomain_statistics!(m, state, aux ,t, ss.statistical_model)
    ∂b∂ρ = - _grav/gm.ρ

    dry.∂e_int∂z
    dry.∂q_tot∂z
    cloudy.∂e_int∂z
    cloudy.∂q_tot∂z

    ρ_i = gm_a.p0/(dry.T*dry.R_m) 
    ∂b∂z_dry = - ∂b∂ρ*ρ_i*( 1/( (1-dry.q_tot)*_cv_d*dry.T + dry.q_vap *_cv_v * dry.T) * dry.∂e_int∂z 
                        + (_R_d/dry.R_m)*(ε_v-1)*dry.∂q_tot∂z - gm_d.∇p0/gm_a.p0)

    ρ_i = gm_a.p0/(cloudy.T*cloudy.R_m) 
    ∂b∂z_cloudy = - ∂b∂ρ*ρ_i*(1/( (1-cloudy.q_tot)*_cv_d+cloudy.q_vap*_cv_v + cloudy.q_liq*_cv_l + cloudy.q_ice*_cv_i)/cloudy.T*cloudy.∂e_int∂z  
                            +(_R_d/dry.R_m) * (1/(_cv_v*(cloudy.T-_T0)+_e_int_i0)*cloudy.∂e_int∂z + (ε_v-1)*cloudy.∂q_tot∂z)
                            - gm_d.∇p0/gm_a.p0)
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
    ts = PhaseEquil(ss.param_set ,en.e_int, gm.ρ, en.q_tot)    
    q = PhasePartition(ts)
    _cp_m = cp_m(param_set, q)
    lv = latent_heat_vapor(ts)
    T = air_temperature(ts);
    ql = PhasePartition(ts).q_liq
    θv =  virtual_pottemp(ts)
    θvl =  θv*exp(-(lv*ql)/(cpm*T))

    ∂θv∂e_int   = 1/(((1-cloudy.q_tot)*_cv_d+cloudy.q_vap *_cv_v
              +cloudy.q_liq*_cv_l+ cloudy.q_ice*_cv_i))* θv/T*(1-lv*ql/cpm*T)
    ∂θvl∂e_int   = 1/(((1-cloudy.q_tot)*_cv_d+cloudy.q_vap *_cv_v
               +cloudy.q_liq*_cv_l+ cloudy.q_ice*_cv_i))* θvl/T*(1-lv*ql/cpm*T)
    ∂θv∂qt   = -θv/Tv*(ε_v-1)/I0 
    ∂θvl∂qt  = -θvl/Tv*(ε_v-1)/I0 
    # apply chain-role
    ∂θv∂z    = ∂θv∂e_int*∂I∂z  + ∂θv∂qt*∂qt∂z
    ∂θvl∂z   = ∂θvl∂e_int*∂I∂z + ∂θvl∂qt*∂qt∂z

    ∂θv∂vl = exp((lv*ql)/(_cp_m*T))
    λ_stb = en.cld_frac
    
    N2 = _grav/θv*( (1-λ_stb)*∂θv∂z + λ_stb*∂θvl∂z*∂θv∂vl)

    return ∂b∂z, N2
end;

function gradient_Richardson_number(∂b∂z, Shear, minval)
    Grad_Ri = min(∂b∂z/max(Shear, eps(FT)), minval)
    return Grad_Ri
end;

function turbulent_Prandtl_number(Pr_n, Grad_Ri, obukhov_length)
    if unstable(obukhov_length)
      Pr_z = Pr_n
    else
      Pr_z = Pr_n*(2*Grad_Ri/(1+(FT(53)/FT(13))*Grad_Ri -sqrt( (1+(FT(53)/FT(130))*Grad_Ri)^2 - 4*Grad_Ri)))
    end
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


