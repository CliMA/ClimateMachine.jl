function nondimensional_exchange_functions(
    ss::SingleStack{FT, N},
    m::EntrainmentDetrainment,
    state::Vars,
    aux::Vars,
    t::Real,
    i::Int,
) where {FT, N}

    # Alias convention:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft
    gm_a = aux
    up_a = aux.edmf.updraft

    # precompute vars
    _grav    = FT(grav(ss.param_set))
    ρinv     = 1/gm.ρ
    up_area  = up[i].ρa*ρinv
    w_up     = up[i].ρau[3]/up[i].ρa
    en_area  = 1-sum([up[j].ρa for j in 1:N])*ρinv
    w_en     = (gm.ρu[3]-sum([up[j].ρau[3] for j in 1:N]))*ρinv
    b_up     = up_a[i].buoyancy
    b_en     = (gm_a.buoyancy-sum([ρinv*up[j].ρa/ρinv*up_a[j].buoyancy for j in 1:N]))
    en_ρe    = (gm.ρe-sum([up[j].ρae for j in 1:N]))/en_area
    en_ρu    = (gm.ρu-sum([up[j].ρae for j in 1:N]))/en_area
    e_pot    = _grav * gm_a.z
    en_e_int = internal_energy(gm.ρ, en_ρe, en_ρu, e_pot)
    en_q_tot = (gm.ρq_tot-sum([up[j].ρaq_tot for j in 1:N]))*ρinv
    up_e_int = internal_energy(gm.ρ, up[i].ρae/up_area, up[i].ρau/up_area, e_pot)
    sqrt_tke = sqrt(abs(en.ρatke)*ρinv/en_area)

    # yair check if I can pass ts_up and ts_en from entr_detr.jl instead of recomputing here
    # ts_up    = PhaseEquil(ss.param_set ,up_e_int, gm.ρ, up[i].ρaq_tot/up[i].ρa)
    # q_con_up = condensate(ts_up)
    # RH_up    = relative_humidity(ts_up)
    # ts_en    = PhaseEquil(ss.param_set ,en_e_int, gm.ρ, en_q_tot)
    # q_con_en = condensate(ts_en)
    # RH_en    = relative_humidity(ts_en)

    q_con_up = FT(0)
    RH_up = FT(0)
    q_con_en = FT(0)
    RH_en = FT(0)
    dw       = max(w_up - w_en,1e-4)
    db       = b_up - b_en

    if RH_up==1.0 || RH_en==1.0
      c_δ = m.c_δ
    else
      c_δ = 0.0
    end
    # compute dry and moist aux functions
    D_ε = m.c_ε/(1+exp(-db/dw/m.μ_0*(m.χ - up_area/(up_area+en_area))))
    D_δ = m.c_ε/(1+exp( db/dw/m.μ_0*(m.χ - up_area/(up_area+en_area))))
    M_δ = m.c_δ*( max((RH_up^m.β-RH_en^m.β),0.0) )^(1/m.β)
    M_ε = m.c_δ*( max((RH_en^m.β-RH_up^m.β),0.0) )^(1/m.β)
    return D_ε ,D_δ ,M_δ ,M_ε
end;

