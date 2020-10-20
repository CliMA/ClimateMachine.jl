#### thermo states for subdomains

using KernelAbstractions: @print

export new_thermo_state_up,
    new_thermo_state_en,
    recover_thermo_state_all,
    recover_thermo_state_up,
    recover_thermo_state_en

####
#### Interface
####

"""
    new_thermo_state_up(bl, state, aux)

Updraft thermodynamic states given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! note
    This method calls saturation adjustment for
    EquilMoist models.

# TODO:
 We need to determine a strategy for when we call
 `recover_thermo_state` vs. `new_thermo_state`.
"""
new_thermo_state_up(
    bl::AtmosModel,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState = recover_thermo_state(bl, state, aux),
) = new_thermo_state_up(bl, bl.moisture, state, aux, ts)

"""
    new_thermo_state_en(bl, state, aux)

Environment thermodynamic state given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! note
    This method calls saturation adjustment for
    EquilMoist models.

# TODO:
 We need to determine a strategy for when we call
 `recover_thermo_state` vs. `new_thermo_state`.
"""
new_thermo_state_en(
    bl::AtmosModel,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState = recover_thermo_state(bl, state, aux),
) = new_thermo_state_en(bl, bl.moisture, state, aux, ts)

"""
    recover_thermo_state_all(bl, state, aux)

Recover NamedTuple of all thermo states
"""
function recover_thermo_state_all(bl, state, aux)
    ts = recover_thermo_state(bl, state, aux)
    return (
        gm = ts,
        en = recover_thermo_state_en(bl, bl.moisture, state, aux, ts),
        up = recover_thermo_state_up(bl, bl.moisture, state, aux, ts),
    )
end

"""
    recover_thermo_state_up(bl, state, aux)

Recover the updraft thermodynamic states given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! note
    This method does _not_ call saturation adjustment for
    EquilMoist models. Instead, it uses the temperature
    in the aux state to construct the thermodynamic state.
    This method assumes that the temperature has been
    previously computed from a new thermodynamic state
    and stored in `aux`.
"""
recover_thermo_state_up(bl, state, aux) =
    recover_thermo_state_up(bl, bl.moisture, state, aux)

"""
    recover_thermo_state_en(bl, state, aux)

Recover the environment thermodynamic state given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! note
    This method does _not_ call saturation adjustment for
    EquilMoist models. Instead, it uses the temperature
    in the aux state to construct the thermodynamic state.
    This method assumes that the temperature has been
    previously computed from a new thermodynamic state
    and stored in `aux`.
"""
recover_thermo_state_en(bl, state, aux) =
    recover_thermo_state_en(bl, bl.moisture, state, aux)

####
#### Implementation
####

function new_thermo_state_up(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
)
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft
    p = air_pressure(ts)

    # compute thermo state for updrafts
    ts_up = vuntuple(N_up) do i
        ρa_up = up[i].ρa
        ρaθ_liq_up = up[i].ρaθ_liq
        ρaq_tot_up = up[i].ρaq_tot
        θ_liq_up = ρaθ_liq_up / ρa_up
        q_tot_up = ρaq_tot_up / ρa_up

        PhaseEquil_pθq(m.param_set, p, θ_liq_up, q_tot_up)
    end
    return ts_up
end

function new_thermo_state_en(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
)
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft

    # diagnose environment thermo state
    ρ_inv = 1 / state.ρ
    p = air_pressure(ts)
    θ_liq = liquid_ice_pottemp(ts)
    q_tot = total_specific_humidity(ts)
    a_en = environment_area(state, aux, N_up)
    θ_liq_en = (θ_liq - sum(vuntuple(j -> up[j].ρaθ_liq * ρ_inv, N_up))) / a_en
    q_tot_en = (q_tot - sum(vuntuple(j -> up[j].ρaq_tot * ρ_inv, N_up))) / a_en
    a_min = m.turbconv.subdomains.a_min
    a_max = m.turbconv.subdomains.a_max
    if !(0 <= θ_liq_en)
        @print("θ_liq_en = ", θ_liq_en, "\n")
        error("Environment θ_liq_en out-of-bounds in new_thermo_state_en")
    end
    if !(0 <= q_tot_en <= 1)
        @print("q_tot_en = ", q_tot_en, "\n")
        error("Environment q_tot_en out-of-bounds in new_thermo_state_en")
    end
    ts_en = PhaseEquil_pθq(m.param_set, p, θ_liq_en, q_tot_en)
    return ts_en
end

function recover_thermo_state_up(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    N_up = n_updrafts(m.turbconv)
    ts_up = vuntuple(N_up) do i
        recover_thermo_state_up_i(m, state, aux, i, ts)
    end
    return ts_up
end

recover_thermo_state_up_i(m, state, aux, i_up, ts) =
    recover_thermo_state_up_i(m, m.moisture, state, aux, i_up, ts)
function recover_thermo_state_up_i(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    i_up,
    ts::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    FT = eltype(state)
    param_set = m.param_set
    up = state.turbconv.updraft

    p = air_pressure(ts)
    T_up_i = aux.turbconv.updraft[i_up].T
    q_tot_up_i = up[i_up].ρaq_tot / up[i_up].ρa
    ρ_up_i = air_density(param_set, T_up_i, p, PhasePartition(q_tot_up_i))
    q_up_i =
        PhasePartition_equil(param_set, T_up_i, ρ_up_i, q_tot_up_i, PhaseEquil)
    e_int_up_i = internal_energy(param_set, T_up_i, q_up_i)
    return PhaseEquil{FT, typeof(param_set)}(
        param_set,
        e_int_up_i,
        ρ_up_i,
        q_up_i.tot,
        T_up_i,
    )
end


function recover_thermo_state_en(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    FT = eltype(state)
    param_set = m.param_set
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft

    p = air_pressure(ts)
    T_en = aux.turbconv.environment.T
    ρ_inv = 1 / state.ρ
    q_tot = total_specific_humidity(ts)
    a_en = environment_area(state, aux, N_up)
    q_tot_en = (q_tot - sum(vuntuple(i -> up[i].ρaq_tot, N_up)) * ρ_inv) / a_en
    ρ_en = air_density(param_set, T_en, p, PhasePartition(q_tot_en))
    q_en = PhasePartition_equil(param_set, T_en, ρ_en, q_tot_en, PhaseEquil)
    e_int_en = internal_energy(param_set, T_en, q_en)
    return PhaseEquil{FT, typeof(param_set)}(
        param_set,
        e_int_en,
        ρ_en,
        q_en.tot,
        T_en,
    )
end
