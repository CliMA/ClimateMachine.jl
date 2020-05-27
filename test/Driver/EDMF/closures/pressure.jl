#### Pressure model kernels

function perturbation_pressure(
    ss::SingleStack{FT, N},
    m::PerturbationPressureModel,
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
    up_d = source.edmf.updraft.diffusive

    ρinv = 1/gm.ρ
    en_area = 1-sum([up[i].ρa for i in 1:N])*ρinv
    w_env = (gm.ρu[3]-sum([up[i].ρau[3] for i in 1:N]))*ρinv
    w_up = up[i].ρu[3] *ρinv

    nh_press_buoy    = - up[i].ρa * up_a[i].buoyancy * m.α_b
    nh_pressure_adv  = up[i].ρa * m.α_a * w_i*up_d.∇u[3]
    nh_pressure_drag = - ρa_k * m.α_d * (w_up - w_env)*abs(w_up - w_env)/max(up_a[i].upd_top, 500.0)

    dpdz = nh_press_buoy + nh_pressure_adv + nh_pressure_drag
    dpdz_tke_i = (w_up - w_env)*dpdz

    return dpdz, dpdz_tke_i
end;