#### Entrainment-Detrainment kernels

function entr_detr(
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
    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft
    
    fill!(m.Λ, 0)
    # precompute vars
    ρinv = 1/gm.ρ
    up_area = up[i].ρa/gm.ρ
    w_up = up[i].ρau[3]/up[i].ρa
    a_en = (1-sum([up[j].ρa*ρinv for j in 1:N]))
    w_en = (gm.ρu[3]-sum([up[j].ρau[3] for j in 1:N]))*ρinv
    b_up = up_a[i].buoyancy
    b_en = (gm_a.buoyancy-sum([ρinv*up[j].ρa/ρinv*up_a[j].buoyancy for j in 1:N]))
    en_e_int = (gm.ρe_int-sum([up[j].ρae_int for j in 1:N]))*ρinv
    en_q_tot = (gm.ρq_tot-sum([up[j].ρaq_tot for j in 1:N]))*ρinv
    sqrt_tke = sqrt(abs(en.ρatke)*ρinv/a_en)
    ts_up = PhaseEquil(ss.param_set ,up[i].ρae_int/up[i].ρa, gm.ρ, up[i].ρaq_tot/up[i].ρa)
    q_con_up = condensate(ts_up)
    ts_en = PhaseEquil(ss.param_set ,en_e_int, gm.ρ, en_q_tot)
    q_con_en = condensate(ts_en)

    dw = max(w_up - w_en,FT(1e-4))
    db = b_up - b_en

    if q_con_up*q_con_en>FT(1e-12)
      c_δ = m.c_δ
    else
      c_δ = 0
    end
    # compute dry and moist nondimentional exchange functions
    # D_ε ,D_δ ,M_δ ,M_ε = nondimensional_exchange_functions(ss ,m, state, aux, t, i)
    # YAIR - bug 1 
    D_ε = FT(1)
    D_δ = FT(1)
    M_δ = FT(1)
    M_ε = FT(1) 

    m.Λ[1] = abs(db/dw)
    m.Λ[2] = m.c_λ*abs(db/(sqrt_tke+sqrt(eps(FT))))
    lower_bound = FT(0.1)
    upper_bound = FT(0.0005)
    λ = lamb_smooth_minimum(m.Λ,lower_bound, upper_bound)

    # compute entrainment/detrainmnet components
    εt = 2*up_area*m.c_t*sqrt_tke/(w_up*up_area*up_a[i].updraft_top)
    ε = λ/w_up*(D_ε + M_ε)
    δ = λ/w_up*(D_δ + M_δ)
    return ε, δ, εt
end;