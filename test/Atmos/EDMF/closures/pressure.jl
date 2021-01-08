#### Pressure model kernels

function perturbation_pressure(bl::AtmosModel{FT}, args, env) where {FT}
    dpdz = ntuple(n_updrafts(bl.turbconv)) do i
        perturbation_pressure(bl, bl.turbconv.pressure, args, env, i)
    end
    return dpdz
end

"""
    perturbation_pressure(
        m::AtmosModel{FT},
        press::PressureModel,
        args,
        env,
        i,
    ) where {FT}

Returns the value of perturbation pressure gradient
for updraft i following He et al. (JAMES, 2020), given:

 - `m`, an `AtmosModel`
 - `press`, a `PressureModel`
 - `args`, top-level arguments
 - `env`, NamedTuple of environment variables
 - `i`, index of the updraft
"""
function perturbation_pressure(
    m::AtmosModel{FT},
    press::PressureModel,
    args,
    env,
    i,
) where {FT}
    @unpack state, diffusive, aux = args
    # Alias convention:
    up = state.turbconv.updraft
    up_aux = aux.turbconv.updraft
    up_dif = diffusive.turbconv.updraft

    w_up_i = fix_void_up(up[i].ρa, up[i].ρaw / up[i].ρa, FT(0))

    nh_press_buoy = press.α_b * up_aux[i].buoyancy
    nh_pressure_adv = -press.α_a * w_up_i * up_dif[i].∇w[3]
    nh_pressure_drag =
        press.α_d * (w_up_i - env.w) * abs(w_up_i - env.w) / press.H_up

    dpdz = nh_press_buoy + nh_pressure_adv + nh_pressure_drag

    return dpdz
end;
