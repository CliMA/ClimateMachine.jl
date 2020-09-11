#### Pressure model kernels

"""
    perturbation_pressure(
        m::AtmosModel{FT},
        press::PressureModel,
        state::Vars,
        diffusive::Vars,
        aux::Vars,
        t::Real,
        env,
        i,
    ) where {FT}
Returns the value of perturbation pressure gradient
for updraft i, as well as the corresponding source term
in the environmental TKE budget following He et al. (JAMES, 2020),
given:
 - `m`, an `AtmosModel`
 - `press`, a `PressureModel`
 - `state`, state variables
 - `diffusive`, additional variables
 - `aux`, auxiliary variables
 - `t`, the time
 - `env`, NamedTuple of environment variables
 - `i`, index of the updraft
"""
function perturbation_pressure(
    m::AtmosModel{FT},
    press::PressureModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    env,
    i,
) where {FT}

    # Alias convention:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    up_a = aux.turbconv.updraft
    up_d = diffusive.turbconv.updraft

    ρinv = 1 / gm.ρ
    N_up = n_updrafts(m.turbconv)
    w_up = up[i].ρaw / up[i].ρa

    nh_press_buoy = press.α_b * up_a[i].buoyancy
    nh_pressure_adv = -press.α_a * w_up * up_d[i].∇w[3]
    nh_pressure_drag =
        press.α_d * (w_up - env.w) * abs(w_up - env.w) / press.H_up

    dpdz = nh_press_buoy + nh_pressure_adv + nh_pressure_drag
    dpdz_tke_i = up[i].ρa * (w_up - env.w) * dpdz

    return dpdz, dpdz_tke_i
end;
