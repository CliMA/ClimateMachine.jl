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
    new_thermo_state(compressibility_model(atmos), atmos, state, aux)

new_thermo_state(::Compressible, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state(
        atmos,
        energy_model(atmos),
        moisture_model(atmos),
        state,
        aux,
    )
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
    new_thermo_state(compressibility_model(atmos), atmos, state, aux)

recover_thermo_state(
    ::Compressible,
    atmos::AtmosModel,
    state::Vars,
    aux::Vars,
) = new_thermo_state(
    atmos,
    energy_model(atmos),
    moisture_model(atmos),
    state,
    aux,
)

recover_thermo_state(::Anelastic1D, atmos::AtmosModel, state::Vars, aux::Vars) =
    new_thermo_state_anelastic(atmos, state, aux)

function new_thermo_state(
    atmos::AtmosModel,
    energy::TotalEnergyModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    e_int = internal_energy(atmos, state, aux)
    return TD.PhaseDry(param_set, e_int, state.ρ)
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::TotalEnergyModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    e_int = internal_energy(atmos, state, aux)
    param_set = parameter_set(atmos)
    return TD.PhaseEquil_ρeq(
        param_set,
        state.ρ,
        e_int,
        state.moisture.ρq_tot / state.ρ,
        moist.maxiter,
        moist.tolerance,
    )
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::TotalEnergyModel,
    moist::NonEquilMoist,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    e_int = internal_energy(atmos, state, aux)
    q = TD.PhasePartition(
        state.moisture.ρq_tot / state.ρ,
        state.moisture.ρq_liq / state.ρ,
        state.moisture.ρq_ice / state.ρ,
    )

    return TD.PhaseNonEquil{eltype(state), typeof(param_set)}(
        param_set,
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
    param_set = parameter_set(atmos)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    return TD.PhaseDry_ρθ(param_set, state.ρ, θ_liq_ice)
end

function new_thermo_state(
    atmos::AtmosModel,
    energy::θModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    param_set = parameter_set(atmos)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    return TD.PhaseEquil_ρθq(
        param_set,
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
    param_set = parameter_set(atmos)
    θ_liq_ice = state.energy.ρθ_liq_ice / state.ρ
    q = TD.PhasePartition(
        state.moisture.ρq_tot / state.ρ,
        state.moisture.ρq_liq / state.ρ,
        state.moisture.ρq_ice / state.ρ,
    )

    return TD.PhaseNonEquil{eltype(state), typeof(param_set)}(
        param_set,
        state.ρ,
        θ_liq_ice,
        q,
    )
end
