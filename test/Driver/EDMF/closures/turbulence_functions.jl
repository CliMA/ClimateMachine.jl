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
    δ::FT,
    εt::FT,
) where {FT, N}
    # think how to call subdomain statistics here to get cloudy and dry values of T if you nee them
    # buoyancy gradients via chain-role
    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft
    en_d = state.edmf.environment.diffusive

    _cv_d::FT    = cv_d(param_set)
    _cv_v::FT    = cp_v(param_set)
    _cv_l::FT    = cp_l(param_set)
    _cv_i::FT    = cp_i(param_set)
    _T0::FT      = T_0(param_set)
    e_int_i0::FT = e_int_i0(param_set)
    ϵ_v::FT      = 1 / molmass_ratio(param_set)
    _grav::FT        = grav(param_set)
    _R_d::FT     = R_d(param_set)

    cloudy, dry = compute_subdomain_statistics!(m, state, aux ,t, m.statistical_model) # can I call the microphyscis model here?     
    ∂b∂ρ = - _grav/gm.ρ
    #                  <-------- ∂ρ∂T -------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_dry    = - _R_d*gm_a.p_0/(dry.R_m*dry.T*dry.T)/((1-dry.q_tot)*_cv_d+dry.q_vap *_cv_v)
    #                  <-------- ∂ρ∂T --------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_cloudy = - (_R_d*gm_a.p_0/(cloudy.R_m*cloudy.T*cloudy.T)/((1-cloudy.q_tot)*_cv_d+cloudy.q_vap *_cv_v+cloudy.q_liq*_cv_l+ cloudy.q_ice*_cv_i)
                       + gm_a.p_0/(cloudy.R_m*cloudy.R_m*cloudy.T)*ϵ_v*_R_d/(_cv_v*(cloudy.T-_T0)+_e_int_i0) )
    #                    <----- ∂ρ∂Rm ------->*<------- ∂Rm∂e_int ---------->

    ∂ρ∂e_int = (en_a.cld_frac * ∂ρ∂e_int_cloudy + (1-en_a.cld_frac) * ∂ρ∂e_int_dry)
    ∂ρ∂q_tot = _R_d*gm_a.p_0/(R_m*R_m*T)
    # apply chain role
    ∂b∂z_e_int = ∂b∂ρ * ∂ρ∂e_int * ∂e_int∂z
    ∂b∂z_q_tot = ∂b∂ρ * ∂ρ∂q_tot * ∂q_tot∂z
    return ∂b∂z_e_int, ∂b∂z_q_tot
end;

function gradient_Richardson_number(∂b∂z_e_int, TKE_Shear, ∂b∂z_q_tot, minval)
    Grad_Ri = min(∂b∂z_e_int/max(TKE_Shear, eps(FT)) + ∂b∂z_q_tot/max(TKE_Shear, eps(FT)) , minval)
    return Grad_Ri
end;

function turbulent_Prandtl_number(Pr_n, Grad_Ri, obukhov_length, a_empirical, b_empirical, c_empirical)
    if unstable(obukhov_length)
      Pr_z = Pr_n
    else
      Pr_z = Pr_n*(2*Grad_Ri/
                        (1+(a_empirical/b_empirical)*Grad_Ri -sqrt( (1+(a_empirical/c_empirical)*Grad_Ri)^2 - 4*Grad_Ri ) ) )
    end
    return Pr_z
end;

function surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, oblength::FT) where FT<:Real
  c_star1 = -flux1/ustar
  c_star2 = -flux2/ustar
  if oblength < 0
    return 4 * c_star1 * c_star2 * (1 - 8.3 * zLL/oblength)^(-2/3)
  else
    return 4 * c_star1 * c_star2
  end
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
    windspeed_min = FT(0.01)
  return max(hypot(gm.u, gm.v), windspeed_min)
end;

function compute_MO_len(k_Karman::FT, ustar::FT, bflux::FT) where {FT<:Real, PS}
  return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / k_Karman
end;

function compute_blux(g::FT, ϵ_v::FT, bflux::FT) where {FT<:Real, PS}
  return (g * ((8.0e-3 + (ϵ_v-1)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (ϵ_v-1) * 22.45e-3))))
end;

