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
for updraft i following He et al. (JAMES, 2020), given:
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
    up_aux = aux.turbconv.updraft
    up_dif = diffusive.turbconv.updraft

    N_up = n_updrafts(m.turbconv)
    w_up_i = up[i].ρaw / up[i].ρa

    nh_press_buoy = press.α_b * up_aux[i].buoyancy
    nh_pressure_adv = -press.α_a * w_up_i * up_dif[i].∇w[3]
    nh_pressure_drag =
        press.α_d * (w_up_i - env.w) * abs(w_up_i - env.w) / press.H_up

    dpdz = nh_press_buoy + nh_pressure_adv + nh_pressure_drag

    return dpdz
end;
