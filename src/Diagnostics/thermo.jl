using ..Atmos
using ..Atmos: MoistureModel

# Helpers to gather the thermodynamic variables across the DG grid
#

function vars_thermo(atmos::AtmosModel, FT)
    @vars begin
        temp::FT
        pres::FT
        θ_dry::FT
        e_int::FT
        h_tot::FT
        h_int::FT

        moisture::vars_thermo(atmos.moisture, FT)
    end
end
vars_thermo(::MoistureModel, FT) = @vars()
function vars_thermo(m::EquilMoist, FT)
    @vars begin
        q_liq::FT
        q_ice::FT
        q_vap::FT
        θ_vir::FT
        θ_liq_ice::FT
    end
end
num_thermo(bl, FT) = varsize(vars_thermo(bl, FT))
thermo_vars(bl, array) = Vars{vars_thermo(bl, eltype(array))}(array)

# compute thermodynamic variables visitor function, to use with `@visitQ`
function compute_thermo!(atmos::AtmosModel, state, aux, thermo)
    e_tot = state.ρe / state.ρ
    ts = thermo_state(atmos, state, aux)
    e_int = internal_energy(ts)

    thermo.temp = air_temperature(ts)
    thermo.pres = air_pressure(ts)
    thermo.θ_dry = dry_pottemp(ts)
    thermo.e_int = e_int

    thermo.h_tot = total_specific_enthalpy(ts, e_tot)
    thermo.h_int = specific_enthalpy(ts)

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
    thermo.moisture.θ_vir = virtual_pottemp(ts)
    thermo.moisture.θ_liq_ice = liquid_ice_pottemp(ts)

    return nothing
end
