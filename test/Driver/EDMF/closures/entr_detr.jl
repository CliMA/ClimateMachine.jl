#### Entrainment-Detrainment kernels

function entr_detr(
    ss::SingleStack{FT, N},
    m::EntrainmentDetrainment,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    i::Int
) where {FT, N}

    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft

    fill!(m.Λ, 0)
    # precompute vars
    ρinv = 1/gm.ρ
    up_area = up[i].ρa/gm.ρ
    w_up = up[i].u[3]
    w_en = (gm.ρu[3]-sum([up[j].ρau[3] for j in 1:N]))*ρinv
    b_up = up[i].buoyancy
    b_en = (gm_a.buoyancy-sum([ρinv*up[j].ρa*up_a[j].buoyancy for j in 1:N]))
    sqrt_tke = sqrt(max(en.tke,0))
    ts_up = PhaseEquil(ss.param_set ,up[i].e_int, up[i].ρ, up[i].q_tot)
    RH_up = relative_humidity(ts_up)
    ts_en = PhaseEquil(ss.param_set ,en_e_int, gm.ρ, en_q_tot)
    RH_en = relative_humidity(ts_en)
    dw = max(w_up - w_en,FT(1e-4))
    db = b_up - b_en

    if RH_up==1.0 || RH_en==1.0
      c_δ = m.c_δ
    else
      c_δ = 0
    end
    # compute dry and moist nondimentional exchange functions
    D_ϵ ,D_δ ,M_δ ,M_ϵ = nondimensional_exchange_functions(ss ,m, state, diffusive, aux, t, direction , i)

    m.Λ[1] = abs(db/dw)
    m.Λ[2] = m.c_λ*abs(db/(sqrt_tke+sqrt(eps(FT))))
    lower_bound = FT(0.1)
    upper_bound = FT(0.0005)
    λ = lamb_smooth_minimum(m.Λ,lower_bound, upper_bound)

    # compute entrainment/detrainmnet components
    εt = 2*up_area*m.c_t*sqrt_tke/(w_up*up_area*up_a[i].cloud.updraft_top)
    ε = λ/w_up*(D_ϵ + M_ϵ)
    δ = λ/w_up*(D_δ + M_δ)
    return εt, ε, δ
end;