#### thermodynamics

export new_thermo_state_anelastic, recover_thermo_state_anelastic

"""
    new_thermo_state_anelastic(atmos::AtmosModel, state::Vars, aux::Vars)

Create a new thermodynamic state, based on the `state`, and _not_
the `aux` state.

!!! note
    This method calls the iterative saturation adjustment
    procedure for EquilMoist models.
"""
new_thermo_state_anelastic(atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state_anelastic(atmos, atmos.energy, atmos.moisture, state, aux)

"""
    recover_thermo_state_anelastic(atmos::AtmosModel, state::Vars, aux::Vars)

An atmospheric thermodynamic state.

!!! warn
    For now, we are directly calling new_thermo_state_anelastic to avoid
    inconsistent aux states in kernels where the aux states are
    out of sync with the boundary state.

# TODO: Define/call `recover_thermo_state_anelastic` when it's safely implemented
  (see https://github.com/CliMA/ClimateMachine.jl/issues/1648)
"""
recover_thermo_state_anelastic(atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state_anelastic(atmos, atmos.energy, atmos.moisture, state, aux)

function new_thermo_state_anelastic(
    atmos::AtmosModel,
    energy::EnergyModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    e_int = internal_energy(atmos, state, aux)
    p = aux.ref_state.p
    return PhaseDry_pe(param_set, p, e_int)
end

function new_thermo_state_anelastic(
    atmos::AtmosModel,
    energy::EnergyModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    e_int = internal_energy(atmos, state, aux)
    p = aux.ref_state.p
    ρ = density(atmos, state, aux)
    return PhaseEquil_peq(
        param_set,
        p,
        e_int,
        state.moisture.ρq_tot / ρ,
        moist.maxiter,
        moist.tolerance,
    )
end

function new_thermo_state_anelastic(
    atmos::AtmosModel,
    energy::EnergyModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    e_int = internal_energy(atmos, state, aux)
    p = aux.ref_state.p
    ρ = density(atmos, state, aux)
    q = PhasePartition(
        state.moisture.ρq_tot / ρ,
        state.moisture.ρq_liq / ρ,
        state.moisture.ρq_ice / ρ,
    )

    return PhaseNonEquil_peq(param_set, p, e_int, q)
end
