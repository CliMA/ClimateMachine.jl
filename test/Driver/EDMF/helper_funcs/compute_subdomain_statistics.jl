#### Helper kernels
# these functions should recive as inputs the state vector and provide as outputs
# the cloud fraction in the subdomain as well as the values of
# T, R_m, q_tot, q_vap ,q_liq, q_ice for the cloudy and dry parts of the subdomain.
# I have stored these values in two structures 'dry' and 'cloudy'.
# When using SubdomainMean the subdomain is either cloudy or dry and all values are the same
# while when using LognormalQuadrature the subdomain can be partially cloudy and partially dry and the cloudy and dry values differ

function compute_subdomain_statistics!(
  m::SingleStack{FT,N},
  state::Vars,
  aux::Vars,
  t::Real,
  statistical_model::String, # this need to be a passed argument that determines the type of statistical model to be used
  ) where {FT,N} # need to call micophysics model as well to populate cloudy and dry

  gm_a = aux
  en_a = aux.edmf.environment
  up_a = aux.edmf.updraft
  gm = state
  en = state.edmf.environment
  up = state.edmf.updraft

  ts = PhaseEquil(param_set ,en.e_int, gm.ρ, en.q_tot)
  T = air_temperature(ts)
  q_liq = PhasePartition(ts).liq
  q_ice = PhasePartition(ts).ice
  # here cloudy and dry are indetical as only one will be used based on the value of cld_frac,
  # but I define both as in the  quadrature options to come they will differ and both will be used
  ## I need to find out how to initiate cloudy and dry structures
  if q_liq+q_ice > 0
    cld_frac = 1
    cloudy.q_tot = en.q_tot
    cloudy.T     = air_temperature(ts)
    cloudy.R_m   = gas_constant_air(ts)
    cloudy.q_vap = q_vap
    cloudy.q_liq = q_liq
    cloudy.q_ice = q_ice
    dry.q_tot = en.q_tot
    dry.T     = cloudy.T
    dry.R_m   = cloudy.R_m
    dry.q_vap = cloudy.q_vap
    dry.q_liq = FT(0)
    dry.q_ice = FT(0)
  else
    cld_frac = 0
    dry.q_tot = en.q_tot
    dry.T     = air_temperature(ts)
    dry.R_m   = gas_constant_air(ts)
    dry.q_vap = PhasePartition(ts).vap
    dry.q_liq = FT(0)
    dry.q_ice = FT(0)
    cloudy.q_tot = en.q_tot
    cloudy.T     = dry.T
    cloudy.R_m   = dry.R_m
    cloudy.q_vap = q_vap
    cloudy.q_liq = q_liq
    cloudy.q_ice = q_ice
  end
  return cloudy, dry, cld_frac
end

## the complete coding of this function can wait for a working model with SubdomainMean
# function compute_subdomain_statistics!(
#   m::SingleStack{FT, N, N_quad},
#   state::Vars,
#   aux::Vars,
#   t::Real,
#   statistical_model::LognormalQuadrature,
#   ) where {FT, N, N_quad}

#   gm_a = aux
#   en_a = aux.edmf.environment
#   up_a = aux.edmf.updraft
#   gm = state
#   en = state.edmf.environment
#   up = state.edmf.updraft

#   env_len = 10
#   src_len = 6

#   inner_env = zeros(env_len);
#   outer_env = zeros(env_len);
#   inner_src = zeros(src_len);
#   outer_src = zeros(src_len);
#   i_Sqt = collect(1:src_len);
#   i_SI = collect(1:src_len);
#   i_Sqt_I = collect(1:src_len);
#   i_Sqt_qt = collect(1:src_len);
#   i_SI_I = collect(1:src_len);
#   i_SI_qt = collect(1:src_len);
#   i_ql = collect(1:env_len);
#   i_T = collect(1:env_len);
#   i_I = collect(1:env_len);
#   i_ρ = collect(1:env_len);
#   i_cf = collect(1:env_len);
#   i_qt_cld = collect(1:env_len);
#   i_qt_dry = collect(1:env_len);
#   i_T_cld = collect(1:env_len);
#   i_T_dry = collect(1:env_len);
#   i_rf = collect(1:env_len);

#   if (en.q_tot_cv > eps(FT) && en.e_int_cv > eps(FT) && fabs(en.e_int_q_tot_cv) > eps(FT)
#       && en.q_tot > eps(FT) && sqrt(en.q_tot_cv) < en.q_tot)
#     σ_q = sqrt(log(en.q_tot_cv/en.q_tot/en.q_tot)+1)
#     σ_h = sqrt(log(en.cv_e_int/en.e_int/en.e_int)+1)
#     # Enforce Schwarz's inequality
#     corr = max(min(en.e_int_q_tot_cv/max(σ_h*σ_q, 1e-13),1),-1)
#     sd2_hq = log(corr*sqrt(en.q_tot_cv*en.e_int)/(en.e_int*en.q_tot) + 1.0)
#     σ_cond_θ_q = sqrt(max(σ_h*σ_h - sd2_hq*sd2_hq/σ_q/σ_q, 0.0))
#     μ_q = log(en.q_tot*en.q_tot/sqrt(en.q_tot*en.q_tot + en.q_tot_cv))
#     μ_θ = log(en.e_int*en.e_int/sqrt(en.e_int*en.e_int + en.e_int_cv))
#     # clean outer vectors
#     outer_src = FT(0)*outer_src
#     outer_env = FT(0)*outer_env

#     for j_qt in 1:N_quad
#       qt_hat = exp(μ_q + sqrt2 * σ_q * abscissas[j_q])
#       μ_eint_star = μ_θ + sd2_hq/sd_q/sd_q*(log(qt_hat)-μ_q)
#       # clean innner vectors
#       inner_src = 0.0*inner_src
#       inner_env = 0.0*inner_env
#       for j_I in 1:N_quad
#         e_int_hat = exp(μ_eint_star + sqrt2 * σ_cond_θ_q * abscissas[j_I])
#         ts = PhaseEquil(param_set, en.e_int, gm.ρ, en.q_tot)
#         T = air_temperature(ts)
#         q_liq = PhasePartition(ts).liq
#         q_vap = PhasePartition(ts).vap
#         q_ice = PhasePartition(ts).ice

#         # microphysics rain should apply here to update qt, ql, thetal, T
#         # now collect the inner values of
#         inner_env[i_ql]     += q_liq * weights[j_I] * sqpi_inv
#         inner_env[i_T]      += T     * weights[j_I] * sqpi_inv
#         inner_env[i_I]      += e_int * weights[j_I] * sqpi_inv
#         inner_env[i_ρ]      += ρ     * weights[j_I] * sqpi_inv
#         # rain area fraction
#         if qr_src > 0
#             inner_env[i_rf]     += weights[j_I] * sqpi_inv
#         end
#         # cloudy/dry categories for buoyancy in TKE
#         if q_liq > 0.0
#             inner_env[i_cf]     +=          weights[j_I] * sqpi_inv
#             inner_env[i_qt_cld] += qt_hat * weights[j_I] * sqpi_inv
#             inner_env[i_T_cld]  += T      * weights[j_I] * sqpi_inv
#         else
#             inner_env[i_qt_dry] += qt_hat * weights[j_I] * sqpi_inv
#             inner_env[i_T_dry]  += T      * weights[j_I] * sqpi_inv
#         end
#         # products for variance and covariance source terms
#         inner_src[i_Sqt]    += -qr_src                 * weights[j_I] * sqpi_inv
#         inner_src[i_SI]     +=  eint_rain_src          * weights[j_I] * sqpi_inv
#         inner_src[i_Sqt_I]  += -qr_src         * e_int * weights[j_I] * sqpi_inv
#         inner_src[i_SI_I]   +=  eint_rain_src  * e_int * weights[j_I] * sqpi_inv
#         inner_src[i_Sqt_qt] += -qr_src         * q_tot * weights[j_I] * sqpi_inv
#         inner_src[i_SI_qt]  +=  eint_rain_src  * q_tot * weights[j_I] * sqpi_inv
#       end

#       outer_env[i_ql]     += inner_env[i_ql]     * weights[j_qt] * sqpi_inv
#       outer_env[i_T]      += inner_env[i_T]      * weights[j_qt] * sqpi_inv
#       outer_env[i_I]      += inner_env[i_I]      * weights[j_qt] * sqpi_inv
#       outer_env[i_ρ]      += inner_env[i_ρ]      * weights[j_qt] * sqpi_inv
#       outer_env[i_cf]     += inner_env[i_cf]     * weights[j_qt] * sqpi_inv
#       outer_env[i_qt_cld] += inner_env[i_qt_cld] * weights[j_qt] * sqpi_inv
#       outer_env[i_qt_dry] += inner_env[i_qt_dry] * weights[j_qt] * sqpi_inv
#       outer_env[i_T_cld]  += inner_env[i_T_cld]  * weights[j_qt] * sqpi_inv
#       outer_env[i_T_dry]  += inner_env[i_T_dry]  * weights[j_qt] * sqpi_inv
#       outer_env[i_rf]     += inner_env[i_rf]     * weights[j_qt] * sqpi_inv

#       outer_src[i_Sqt]    += inner_src[i_Sqt]    * weights[j_qt] * sqpi_inv
#       outer_src[i_SI]     += inner_src[i_SI]     * weights[j_qt] * sqpi_inv
#       outer_src[i_Sqt_I]  += inner_src[i_Sqt_I]  * weights[j_qt] * sqpi_inv
#       outer_src[i_SI_I]   += inner_src[i_SI_I]   * weights[j_qt] * sqpi_inv
#       outer_src[i_Sqt_qt] += inner_src[i_Sqt_qt] * weights[j_qt] * sqpi_inv
#       outer_src[i_SI_qt]  += inner_src[i_SI_qt]  * weights[j_qt] * sqpi_inv
#     end

#     en_a.cld_frac = outer_env[i_cf]
#     cloudy.q_tot = outer_env[i_qt_cld]
#     cloudy.T     = outer_env[i_T]
#     cloudy.R_m   = gas_constant_air(ts)
#     cloudy.q_vap = q_vap
#     cloudy.q_liq = q_liq
#     cloudy.q_ice = q_ice
#     dry.q_tot = en.q_tot
#     dry.T     = cloudy.T
#     dry.R_m   = cloudy.R_m
#     dry.q_vap = cloudy.q_vap
#     dry.q_liq = FT(0)
#     dry.q_ice = FT(0)

#     en.cld_frac  =
#     t_cloudy     = outer_env[i_T]
#     q_tot_cloudy = outer_env[i_qt_cld]
#     q_vap_cloudy = outer_env[i_qt_cld] - outer_env[i_ql]
#     q_tot_dry    = outer_env[i_qt_dry]

#     ts = TemperatureSHumEquil(param_set, t_cloudy, p_0, q_tot_cloudy) # revisit this thermo stats
#     θ_cloudy = liquid_ice_pottemp(ts)
#     θ_dry    = dry_pottemp(ts)

#     return Sqt_eint_cov, Sqt_var
#   end
# end