# Convenience wrapper
save_subdomain_temperature!(m, state, aux) =
    save_subdomain_temperature!(m, m.moisture, state, aux)

using KernelAbstractions: @print

"""
    save_subdomain_temperature!(
        m::AtmosModel,
        moist::EquilMoist,
        state::Vars,
        aux::Vars,
    )

Updates the subdomain sensible temperature, given:
 - `m`, an `AtmosModel`
 - `moist`, an `EquilMoist` model
 - `state`, state variables
 - `aux`, auxiliary variables
"""
function save_subdomain_temperature!(
    m::AtmosModel,
    moist::EquilMoist,
    state::Vars,
    aux::Vars,
)
    ts = recover_thermo_state(m, state, aux)
    ts_en = new_thermo_state_en(m, state, aux, ts)

    aux.turbconv.environment.T = air_temperature(ts_en)
    return nothing
end

# No need to save temperature for DryModel.
function save_subdomain_temperature!(
    m::AtmosModel,
    moist::DryModel,
    state::Vars,
    aux::Vars,
) end
