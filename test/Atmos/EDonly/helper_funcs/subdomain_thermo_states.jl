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
    )
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

function new_thermo_state_en(
    m::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
    ts::ThermodynamicState,
)

    # diagnose environment thermo state
    p = pressure(m, state, aux)
    ρ = density(m, state, aux)
    ρ_inv = 1 / ρ
    θ_liq = liquid_ice_pottemp(ts)
    a_en = environment_area(state)
    θ_liq_en = θ_liq
    if !(0 <= θ_liq_en)
        @print("θ_liq = ", θ_liq, "\n")
        @print("θ_liq_en = ", θ_liq_en, "\n")
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
    # diagnose environment thermo state
    ρ = density(m, state, aux)
    p = pressure(m, state, aux)
    ρ_inv = 1 / ρ
    θ_liq = liquid_ice_pottemp(ts)
    q_tot = total_specific_humidity(ts)
    a_en = environment_area(state)
    θ_liq_en = θ_liq
    q_tot_en = q_tot
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
