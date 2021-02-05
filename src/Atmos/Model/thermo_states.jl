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
function new_thermo_state end

# First dispatch on compressibility
new_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(atmos.compressibility, atmos, state, aux)

new_thermo_state(::Compressible, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(atmos, state, aux)
new_thermo_state(::Anelastic1D, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state_anelastic(atmos, state, aux)

"""
    recover_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars)

An atmospheric thermodynamic state.

!!! warn
    For now, we are directly calling new_thermo_state to avoid
    inconsistent aux states in kernels where the aux states are
    out of sync with the boundary state.

# TODO: Define/call `recover_thermo_state` when it's safely implemented
  (see https://github.com/CliMA/ClimateMachine.jl/issues/1648)
"""
function recover_thermo_state end

# First dispatch on compressibility
recover_thermo_state(atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(atmos.compressibility, atmos, state, aux)

recover_thermo_state(::Compressible, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(atmos, state, aux)

recover_thermo_state(::Anelastic1D, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state_anelastic(atmos, state, aux)

function new_thermo_state(
    atmos::AtmosModel,
    energy::EnergyModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    return PhaseDry(atmos.param_set, e_int, state.ρ)
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::EnergyModel,
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
    energy::EnergyModel,
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

function new_thermo_state(
    atmos::AtmosModel,
    energy::θModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    return PhaseDry_ρθ(atmos.param_set, state.ρ, θ_liq_ice)
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::θModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    return PhaseEquil_ρθq(
        atmos.param_set,
        state.ρ,
        θ_liq_ice,
        state.moisture.ρq_tot / state.ρ,
        moist.maxiter,
        moist.tolerance,
    )
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::θModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    q = PhasePartition(
        state.moisture.ρq_tot / state.ρ,
        state.moisture.ρq_liq / state.ρ,
        state.moisture.ρq_ice / state.ρ,
    )

    return PhaseNonEquil{eltype(state), typeof(atmos.param_set)}(
        atmos.param_set,
        state.ρ,
        θ_liq_ice,
        q,
    )
end
