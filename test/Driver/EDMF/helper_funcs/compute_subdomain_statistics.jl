#### Helper kernels

function compute_subdomain_statistics!(
  m::SingleStack{FT,N},
  state::Vars,
  aux::Vars,
  t::Real,
  statistical_model::SubdomainMean,
  ) where {FT,IT}

  gm_a = aux
  en_a = aux.edmf.environment
  up_a = aux.edmf.updraft
  gm = state
  en = state.edmf.environment
  up = state.edmf.updraft

  @unpack params param_set
  @inbounds for k in over_elems_real(grid)
    q_tot = q[:q_tot, k, en]
    ts = ActiveThermoState(param_set, q, tmp, k, en)
    T = air_temperature(ts)
    q_liq = PhasePartition(ts).liq
    q_vap = q_tot - q_liq
    θ = dry_pottemp(ts)
    if q_liq > 0
      tmp[:CF, k] = FT(1)
      tmp[:θ_cloudy, k]     = θ
      tmp[:t_cloudy, k]     = T
      tmp[:q_tot_cloudy, k] = q_tot
      tmp[:q_vap_cloudy, k] = q_vap
    else
      tmp[:CF, k] = FT(0)
      tmp[:θ_dry, k]     = θ
      tmp[:q_tot_dry, k] = q_tot
    end
  end
end



function compute_subdomain_statistics!(
  m::SingleStack{FT,N},
  state::Vars,
  aux::Vars,
  t::Real,
  statistical_model::Lognormal,
  ) where {FT, N}

  gm_a = aux
  en_a = aux.edmf.environment
  up_a = aux.edmf.updraft
  gm = state
  en = state.edmf.environment
  up = state.edmf.updraft

  env_len = 10
  src_len = 6

  inner_env = zeros(env_len);
  outer_env = zeros(env_len);
  inner_src = zeros(src_len);
  outer_src = zeros(src_len);
  i_Sqt = collect(1:src_len);
  i_SH = collect(1:src_len);
  i_Sqt_H = collect(1:src_len);
  i_Sqt_qt = collect(1:src_len);
  i_SH_H = collect(1:src_len);
  i_SH_qt = collect(1:src_len);
  i_ql = collect(1:env_len);
  i_T = collect(1:env_len);
  i_eint = collect(1:env_len);
  i_ρ = collect(1:env_len);
  i_cf = collect(1:env_len);
  i_qt_cld = collect(1:env_len);
  i_qt_dry = collect(1:env_len);
  i_T_cld = collect(1:env_len);
  i_T_dry = collect(1:env_len);
  i_rf = collect(1:env_len);

  e_int_cv


  if (en.q_tot_cv > eps(FT) && en.e_int_cv > eps(FT) && fabs(en.e_int_q_tot_cv) > eps(FT)
      && en.q_tot > eps(FT) && sqrt(en.q_tot_cv) < en.q_tot)
    σ_q = sqrt(log(en.q_tot_cv/en.q_tot/en.q_tot)+1)
    σ_h = sqrt(log(en.cv_e_int/en.e_int/en.e_int)+1)
    # Enforce Schwarz's inequality
    corr = max(min(en.e_int_q_tot_cv/max(σ_h*σ_q, 1e-13),1),-1)
    sd2_hq = log(corr*sqrt(en.q_tot_cv*en.e_int)/(en.e_int*en.q_tot) + 1.0)
    σ_cond_θ_q = sqrt(max(σ_h*σ_h - sd2_hq*sd2_hq/σ_q/σ_q, 0.0))
    μ_q = log(en.q_tot*en.q_tot/sqrt(en.q_tot*en.q_tot + en.q_tot_cv))
    μ_θ = log(en.e_int*en.e_int/sqrt(en.e_int*en.e_int + en.e_int_cv, k, i]))
    # clean outer vectors
    outer_src = 0.0*outer_src
    outer_env = 0.0*outer_env

    for j_qt in 1:m.quadrature_order
      qt_hat = exp(μ_q + sqrt2 * σ_q * abscissas[j_q])
      μ_eint_star = μ_θ + sd2_hq/sd_q/sd_q*(log(qt_hat)-μ_q)
      # clean innner vectors
      inner_src = 0.0*inner_src
      inner_env = 0.0*inner_env
      for j_eint in 1:model.order
        e_int_hat = exp(μ_eint_star + sqrt2 * σ_cond_θ_q * abscissas[j_eint])
        ts = moist_thermo()??
        T = air_temperature(ts)
        q_liq = PhasePartition(ts).liq
        q_vap = q_tot - q_liq
        θ = dry_pottemp(ts)

        # microphysics rain should apply here to update qt, ql, thetal, T
        # now collect the inner values of
        inner_env[i_ql]     += q_liq * weights[j_eint] * sqpi_inv
        inner_env[i_T]      += T     * weights[j_eint] * sqpi_inv
        inner_env[i_eint]   += e_int * weights[j_eint] * sqpi_inv
        inner_env[i_ρ]      += ρ     * weights[j_eint] * sqpi_inv
        # rain area fraction
        if qr_src > 0
            inner_env[i_rf]     += weights[j_eint] * sqpi_inv
        end
        # cloudy/dry categories for buoyancy in TKE
        if q_liq > 0.0
            inner_env[i_cf]     +=          weights[j_eint] * sqpi_inv
            inner_env[i_qt_cld] += qt_hat * weights[j_eint] * sqpi_inv
            inner_env[i_T_cld]  += T      * weights[j_eint] * sqpi_inv
        else
            inner_env[i_qt_dry] += qt_hat * weights[j_eint] * sqpi_inv
            inner_env[i_T_dry]  += T      * weights[j_eint] * sqpi_inv
        end
        # products for variance and covariance source terms
        inner_src[i_Sqt]    += -qr_src                 * weights[j_eint] * sqpi_inv
        inner_src[i_SH]     +=  eint_rain_src          * weights[j_eint] * sqpi_inv
        inner_src[i_Sqt_H]  += -qr_src         * θ     * weights[j_eint] * sqpi_inv
        inner_src[i_SH_H]   +=  eint_rain_src  * θ     * weights[j_eint] * sqpi_inv
        inner_src[i_Sqt_qt] += -qr_src         * q_tot * weights[j_eint] * sqpi_inv
        inner_src[i_SH_qt]  +=  eint_rain_src  * q_tot * weights[j_eint] * sqpi_inv
      end

      outer_env[i_ql]     += inner_env[i_ql]     * weights[j_qt] * sqpi_inv
      outer_env[i_T]      += inner_env[i_T]      * weights[j_qt] * sqpi_inv
      outer_env[i_eint]   += inner_env[i_eint]   * weights[j_qt] * sqpi_inv
      outer_env[i_ρ]      += inner_env[i_ρ]      * weights[j_qt] * sqpi_inv
      outer_env[i_cf]     += inner_env[i_cf]     * weights[j_qt] * sqpi_inv
      outer_env[i_qt_cld] += inner_env[i_qt_cld] * weights[j_qt] * sqpi_inv
      outer_env[i_qt_dry] += inner_env[i_qt_dry] * weights[j_qt] * sqpi_inv
      outer_env[i_T_cld]  += inner_env[i_T_cld]  * weights[j_qt] * sqpi_inv
      outer_env[i_T_dry]  += inner_env[i_T_dry]  * weights[j_qt] * sqpi_inv
      outer_env[i_rf]     += inner_env[i_rf]     * weights[j_qt] * sqpi_inv

      outer_src[i_Sqt]    += inner_src[i_Sqt]    * weights[j_qt] * sqpi_inv
      outer_src[i_SH]     += inner_src[i_SH]     * weights[j_qt] * sqpi_inv
      outer_src[i_Sqt_H]  += inner_src[i_Sqt_H]  * weights[j_qt] * sqpi_inv
      outer_src[i_SH_H]   += inner_src[i_SH_H]   * weights[j_qt] * sqpi_inv
      outer_src[i_Sqt_qt] += inner_src[i_Sqt_qt] * weights[j_qt] * sqpi_inv
      outer_src[i_SH_qt]  += inner_src[i_SH_qt]  * weights[j_qt] * sqpi_inv
    end

    en.cld_frac              = outer_env[i_cf]
    tmp[:t_cloudy, k, i]     = outer_env[i_T]
    tmp[:q_tot_cloudy, k, i] = outer_env[i_qt_cld]
    tmp[:q_vap_cloudy, k, i] = outer_env[i_qt_cld] - outer_env[i_ql]
    tmp[:q_tot_dry, k, i]    = outer_env[i_qt_dry]

    ts = TemperatureSHumEquil(param_set, tmp[:t_cloudy, k], p_0=tmp[:p_0, k, i], tmp[:q_tot_cloudy, k, i])
    tmp[:θ_cloudy, k, i]     = liquid_ice_pottemp(ts)
    tmp[:θ_dry, k, i]        = dry_pottemp(ts)

    return Sqt_eint_cov, Sqt_var
  end
end