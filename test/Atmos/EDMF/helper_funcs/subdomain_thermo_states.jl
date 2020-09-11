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
    ts_gm::ThermodynamicState = recover_thermo_state(bl, state, aux),
) = new_thermo_state_up(bl, bl.moisture, state, aux, ts_gm)

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
    ts_gm::ThermodynamicState = recover_thermo_state(bl, state, aux),
) = new_thermo_state_en(bl, bl.moisture, state, aux, ts_gm)

"""
    recover_thermo_state_all(bl, state, aux)

Recover NamedTuple of all thermo states
"""
function recover_thermo_state_all(bl, state, aux)
    ts_gm = recover_thermo_state(bl, state, aux)
    return (
        gm = ts_gm,
        en = recover_thermo_state_en(bl, bl.moisture, state, aux, ts_gm),
        up = recover_thermo_state_up(bl, bl.moisture, state, aux, ts_gm),
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
    ts_gm::ThermodynamicState,
)
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft
    p = air_pressure(ts_gm)

    # compute thermo state for updrafts
    ts_up = vuntuple(N_up) do i
        ρa_up = up[i].ρa
        ρaθ_liq_up = up[i].ρaθ_liq
        ρaq_tot_up = up[i].ρaq_tot
        θ_liq_up = ρaθ_liq_up / ρa_up
        q_tot_up = ρaq_tot_up / ρa_up

        LiquidIcePotTempSHumEquil_given_pressure(
            m.param_set,
            θ_liq_up,
            p,
            q_tot_up,
        )
    end
    return ts_up
end

function new_thermo_state_en(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts_gm::ThermodynamicState,
)
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft

    # diagnose environment thermo state
    ρinv = 1 / state.ρ
    p = air_pressure(ts_gm)
    θ_liq_gm = liquid_ice_pottemp(ts_gm)
    q_tot_gm = total_specific_humidity(ts_gm)
    a_en = environment_area(state, aux, N_up)
    θ_liq_en =
        (θ_liq_gm - sum(vuntuple(j -> up[j].ρaθ_liq * ρinv, N_up))) / a_en
    q_tot_en =
        (q_tot_gm - sum(vuntuple(j -> up[j].ρaq_tot * ρinv, N_up))) / a_en
    ts_en = LiquidIcePotTempSHumEquil_given_pressure(
        m.param_set,
        θ_liq_en,
        p,
        q_tot_en,
    )
    return ts_en
end

function recover_thermo_state_up(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts_gm::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    N_up = n_updrafts(m.turbconv)
    ts_up = vuntuple(N_up) do i
        recover_thermo_state_up_i(m, state, aux, i, ts_gm)
    end
    return ts_up
end

recover_thermo_state_up_i(m, state, aux, i_up, ts_gm) =
    recover_thermo_state_up_i(m, m.moisture, state, aux, i_up, ts_gm)
function recover_thermo_state_up_i(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    i_up,
    ts_gm::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    FT = eltype(state)
    param_set = m.param_set
    up = state.turbconv.updraft

    p = air_pressure(ts_gm)
    T = aux.turbconv.updraft[i_up].T
    q_tot = up[i_up].ρaq_tot / up[i_up].ρa
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, PhaseEquil)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end


function recover_thermo_state_en(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts_gm::ThermodynamicState = recover_thermo_state(m, state, aux),
)
    FT = eltype(state)
    param_set = m.param_set
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft

    p = air_pressure(ts_gm)
    T = aux.turbconv.environment.T
    ρinv = 1 / state.ρ
    ρaq_tot_en =
        total_specific_humidity(ts_gm) -
        sum(vuntuple(i -> up[i].ρaq_tot, N_up)) * ρinv
    a_en = environment_area(state, aux, N_up)
    q_tot = ρaq_tot_en * ρinv / a_en
    ρ = air_density(param_set, T, p, PhasePartition(q_tot))
    q = PhasePartition_equil(param_set, T, ρ, q_tot, PhaseEquil)
    e_int = internal_energy(param_set, T, q)
    return PhaseEquil{FT, typeof(param_set)}(param_set, e_int, ρ, q.tot, T)
end
