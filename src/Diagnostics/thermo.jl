using ..Atmos
using ..Atmos: MoistureModel

# Helpers to gather the thermodynamic variables across the DG grid
#

function vars_thermo(m::AtmosModel, FT)
    @vars begin
        T::FT
        θ_dry::FT
        θ_vir::FT
        e_int::FT
        h_tot::FT
        h_moi::FT

        moisture::vars_thermo(m.moisture, FT)
    end
end
vars_thermo(::MoistureModel, FT) = @vars()
function vars_thermo(m::EquilMoist, FT)
    @vars begin
        q_liq::FT
        q_ice::FT
        q_vap::FT
        θ_liq_ice::FT
    end
end
num_thermo(bl, FT) = varsize(vars_thermo(bl, FT))
thermo_vars(bl, array) = Vars{vars_thermo(bl, eltype(array))}(array)

# visitor function, to use with `@visitQ`
function compute_thermo!(atmos::AtmosModel, state, aux, thermo)
    e_tot = state.ρe / state.ρ
    ts = thermo_state(atmos, state, aux)
    e_int = internal_energy(ts)

    thermo.T = air_temperature(ts)
    thermo.θ_dry = dry_pottemp(ts)
    thermo.θ_vir = virtual_pottemp(ts)
    thermo.e_int = e_int

    # Moist and total henthalpy
    R_m = gas_constant_air(ts)
    thermo.h_tot = e_tot + R_m * thermo.T
    thermo.h_moi = e_int + R_m * thermo.T

    compute_thermo!(atmos.moisture, state, aux, ts, thermo)

    return nothing
end
function compute_thermo!(::MoistureModel, state, aux, ts, thermo)
    return nothing
end
function compute_thermo!(moist::EquilMoist, state, aux, ts, thermo)
    Phpart = PhasePartition(ts)

    thermo.moisture.q_liq = Phpart.liq
    thermo.moisture.q_ice = Phpart.ice
    thermo.moisture.q_vap = vapor_specific_humidity(ts)
    thermo.moisture.θ_liq_ice = liquid_ice_pottemp(ts)

    return nothing
end
