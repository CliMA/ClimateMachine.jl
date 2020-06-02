function nondimensional_exchange_functions(
    ss::SingleStack{FT, N},
    m::EntrainmentDetrainment,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    i::N
) where {FT, N}

    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft

    # precompute vars
    ρinv = 1/gm.ρ
    up_area = up[i].ρa*ρinv
    w_up = up[i].u[3]
    en_area = 1-sum([up[j].ρa for j in 1:N])*ρinv
    w_en = (gm.ρu[3]-sum([up[j].ρau[3] for j in 1:N]))*ρinv
    b_up = up[i].buoyancy
    b_en = (gm_a.buoyancy-sum([ρinv*up[j].ρa*up_a[j].buoyancy for j in 1:N]))
    en_e_int = (gm_a.ρe_int-sum([up[j].ρae_int for j in 1:N])) * ρinv
    en_q_tot = (gm_a.ρq_tot-sum([up[j].ρaq_tot for j in 1:N])) * ρinv
    en_e_int = (gm.ρe_int-up[i].ρae_int)/(gm.ρ*up_area)
    en_q_tot = (gm.ρq_tot-up[i].ρaq_tot)/(gm.ρ*up_area)
    sqrt_tke = sqrt(max(en.tke,0.0))
    ts_up = PhaseEquil(param_set ,up[i].e_int, up[i].ρ, up[i].q_tot)
    RH_up = relative_humidity(ts_up)
    ts_en = PhaseEquil(param_set ,en_e_int, gm.ρ, en_q_tot)
    RH_en = relative_humidity(ts_en)
    dw = max(w_up - w_en,1e-4)
    db = b_up - b_en

    if RH_up==1.0 || RH_en==1.0
      c_δ = m.c_δ
    else
      c_δ = 0.0
    end
    # compute dry and moist aux functions
    D_ϵ = m.c_ε/(1+exp(-db/dw/m.μ_0*(m.χ - up_area/(up_area+en_area))))
    D_δ = m.c_ε/(1+exp( db/dw/m.μ_0*(m.χ - up_area/(up_area+en_area))))
    M_δ = m.c_δ*( max((RH_up^β-RH_en^m.β),0.0) )^(1/m.β)
    M_ϵ = m.c_δ*( max((RH_en^β-RH_up^m.β),0.0) )^(1/m.β)
    return D_ϵ ,D_δ ,M_δ ,M_ϵ
end;

