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

# TODO: Define/call `recover_thermo_state` when it's safely implemented
  (see https://github.com/CliMA/ClimateMachine.jl/issues/1648)
"""
function recover_thermo_state_all(bl, state, aux)
    ts = new_thermo_state(bl, state, aux)
    return (
        gm = ts,
        en = new_thermo_state_en(bl, bl.moisture, state, aux, ts),
        up = new_thermo_state_up(bl, bl.moisture, state, aux, ts),
    )
end

"""
    recover_thermo_state_up(bl, state, aux, ts = new_thermo_state(bl, state, aux))

Recover the updraft thermodynamic states given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! warn
    Right now we are directly calling new_thermo_state_up to avoid
    inconsistent aux states in kernels where the aux states are
    out of sync with the boundary state.

# TODO: Define/call `recover_thermo_state` when it's safely implemented
  (see https://github.com/CliMA/ClimateMachine.jl/issues/1648)
"""
function recover_thermo_state_up(
    bl,
    state,
    aux,
    ts = new_thermo_state(bl, state, aux),
)
    return new_thermo_state_up(bl, bl.moisture, state, aux, ts)
end

"""
    recover_thermo_state_en(bl, state, aux, ts = recover_thermo_state(bl, state, aux))

Recover the environment thermodynamic state given:
 - `bl`, parent `BalanceLaw`
 - `state`, state variables
 - `aux`, auxiliary variables

!!! warn
    Right now we are directly calling new_thermo_state_up to avoid
    inconsistent aux states in kernels where the aux states are
    out of sync with the boundary state.

# TODO: Define/call `recover_thermo_state` when it's safely implemented
  (see https://github.com/CliMA/ClimateMachine.jl/issues/1648)
"""
function recover_thermo_state_en(
    bl,
    state,
    aux,
    ts = new_thermo_state(bl, state, aux),
)
    return new_thermo_state_en(bl, bl.moisture, state, aux, ts)
end

####
#### Implementation
####

function new_thermo_state_up(
    m::AtmosModel{FT},
    moist::DryModel,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
) where {FT}
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft
    # check if 1D_anelastic
    p = pressure(m, state, aux)

    # compute thermo state for updrafts
    ts_up = vuntuple(N_up) do i
        ρa_up = up[i].ρa
        ρaθ_liq_up = up[i].ρaθ_liq
        θ_liq_up =
            fix_void_up(ρa_up, ρaθ_liq_up / ρa_up, liquid_ice_pottemp(ts))
        PhaseDry_pθ(m.param_set, p, θ_liq_up)
    end
    return ts_up
end

function new_thermo_state_up(
    m::AtmosModel{FT},
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
) where {FT}

    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft
    # check if 1D_anelastic
    p = pressure(m, state, aux)

    # compute thermo state for updrafts
    ts_up = vuntuple(N_up) do i
        ρa_up = up[i].ρa
        ρaθ_liq_up = up[i].ρaθ_liq
        ρaq_tot_up = up[i].ρaq_tot
        θ_liq_up =
            fix_void_up(ρa_up, ρaθ_liq_up / ρa_up, liquid_ice_pottemp(ts))
        q_tot_up = fix_void_up(
            ρa_up,
            ρaq_tot_up / ρa_up,
            total_specific_humidity(ts),
        )
        PhaseEquil_pθq(m.param_set, p, θ_liq_up, q_tot_up)
    end
    return ts_up
end

function new_thermo_state_en(
    m::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
)
    N_up = n_updrafts(m.turbconv)
    up = state.turbconv.updraft
    # check if 1D_anelastic
    ρ = density(m, state, aux)
    p = pressure(m, state, aux)

    # diagnose environment thermo state
    ρ_inv = 1 / ρ
    θ_liq = liquid_ice_pottemp(ts)
    a_en = environment_area(state, N_up)
    ρaθ_liq_up = vuntuple(N_up) do i
        fix_void_up(up[i].ρa, up[i].ρaθ_liq, up[i].ρa*θ_liq)
    end
    θ_liq_en = (θ_liq - sum(vuntuple(j -> ρaθ_liq_up[j] * ρ_inv, N_up))) / a_en
    a_min = m.turbconv.subdomains.a_min
    a_max = m.turbconv.subdomains.a_max
    if !(0 <= θ_liq_en)
        @show(ts)
        @print("ρaθ_liq_up = ", ρaθ_liq_up[Val(1)], "\n")
        @print("θ_liq = ", θ_liq, "\n")
        @print("θ_liq_en = ", θ_liq_en, "\n")
        @print("ρ = ", ρ, "\n")
        @print("p = ", p, "\n")
        @print("ρa_up = ", up[Val(1)].ρa, "\n")
        @print("a_en = ", a_en, "\n")
        @print("a_min = ", a_min, "\n")
        @print("a_max = ", a_max, "\n")
        error("Environment θ_liq_en out-of-bounds in new_thermo_state_en")
    end
    ts_en = PhaseDry_pθ(m.param_set, p, θ_liq_en)
    return ts_en
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

    # check if 1D_anelastic
    ρ = density(m, state, aux)
    p = pressure(m, state, aux)

    # diagnose environment thermo state
    ρ_inv = 1 / ρ
    θ_liq = liquid_ice_pottemp(ts)
    q_tot = total_specific_humidity(ts)
    a_en = environment_area(state, N_up)
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
