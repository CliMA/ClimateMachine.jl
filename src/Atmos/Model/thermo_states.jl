#### thermodynamics

export new_thermo_state, recover_thermo_state

"""
    new_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars)

Create a new thermodynamic state, based on the `state`, and _not_
the `aux` state.

!!! note
    This method calls the iterative saturation adjustment
    procedure for EquilMoist models.
"""
new_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(atmos, atmos.moisture, state, aux)

function new_thermo_state(
    atmos::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    return PhaseDry(atmos.param_set, e_int, state.ρ)
end

function new_thermo_state(
    atmos::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    return PhaseEquil(
        atmos.param_set,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        moist.maxiter,
        moist.tolerance,
    )
end

function new_thermo_state(
    atmos::AtmosModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    q = PhasePartition(
        state.moisture.ρq_tot / state.ρ,
        state.moisture.ρq_liq / state.ρ,
        state.moisture.ρq_ice / state.ρ,
    )

    return PhaseNonEquil{eltype(state), typeof(atmos.param_set)}(
        atmos.param_set,
        e_int,
        state.ρ,
        q,
    )
end

"""
    recover_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars)

Recover the thermodynamic state, based on the `state` _and_
the `aux` state, based on the existing temperature.

!!! note
    This method does _not_ call the iterative saturation adjustment
    procedure.
"""
recover_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    recover_thermo_state(atmos, atmos.moisture, state, aux)

function recover_thermo_state(
    atmos::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    return PhaseDry(atmos.param_set, e_int, state.ρ)
end

function recover_thermo_state(
    atmos::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    return PhaseEquil{eltype(state), typeof(atmos.param_set)}(
        atmos.param_set,
        e_int,
        state.ρ,
        state.moisture.ρq_tot / state.ρ,
        aux.moisture.temperature,
    )
end

function recover_thermo_state(
    atmos::AtmosModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    q = PhasePartition(
        state.moisture.ρq_tot / state.ρ,
        state.moisture.ρq_liq / state.ρ,
        state.moisture.ρq_ice / state.ρ,
    )

    return PhaseNonEquil{eltype(state), typeof(atmos.param_set)}(
        atmos.param_set,
        e_int,
        state.ρ,
        q,
    )
end
