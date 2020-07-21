### Reference state
using DocStringExtensions
using ..TemperatureProfiles
export ReferenceState, NoReferenceState, HydrostaticState

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

"""
    ReferenceState

Hydrostatic reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

vars_state_conservative(m::ReferenceState, FT) = @vars()
vars_state_gradient(m::ReferenceState, FT) = @vars()
vars_state_gradient_flux(m::ReferenceState, FT) = @vars()
vars_state_auxiliary(m::ReferenceState, FT) = @vars()
atmos_init_aux!(
    ::ReferenceState,
    ::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) = nothing

"""
    NoReferenceState <: ReferenceState

No reference state used
"""
struct NoReferenceState <: ReferenceState end

"""
    HydrostaticState{P,T} <: ReferenceState

A hydrostatic state specified by a virtual
temperature profile and relative humidity.

By default, this is a dry hydrostatic reference
state.
"""
struct HydrostaticState{P, FT} <: ReferenceState
    virtual_temperature_profile::P
    relative_humidity::FT
end
function HydrostaticState(
    virtual_temperature_profile::TemperatureProfile{FT},
) where {FT}
    return HydrostaticState{typeof(virtual_temperature_profile), FT}(
        virtual_temperature_profile,
        FT(0),
    )
end

vars_state_auxiliary(m::HydrostaticState, FT) =
    @vars(ρ::FT, p::FT, T::FT, ρe::FT, ρq_tot::FT)


function atmos_init_aux!(
    m::HydrostaticState{P, F},
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) where {P, F}
    z = altitude(atmos, aux)
    T_virt, p = m.virtual_temperature_profile(atmos.param_set, z)
    FT = eltype(aux)
    _R_d::FT = R_d(atmos.param_set)

    # Replace density by computation from pressure
    # ρ = -1/g*dpdz
    ρ = p / (_R_d * T_virt)
    aux.ref_state.ρ = ρ
    aux.ref_state.p = p
    RH = m.relative_humidity
    phase_type = PhaseEquil
    (T, q_pt) = temperature_and_humidity_from_virtual_temperature(
        atmos.param_set,
        T_virt,
        ρ,
        RH,
        phase_type,
    )

    # Update temperature to be exactly consistent with
    # p, ρ, and q_pt
    T = air_temperature_from_ideal_gas_law(atmos.param_set, p, ρ, q_pt)
    q_tot = q_pt.tot
    ts = TemperatureSHumEquil(atmos.param_set, T, ρ, q_tot)

    aux.ref_state.ρq_tot = ρ * q_tot
    aux.ref_state.T = T
    e_kin = F(0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    aux.ref_state.ρe = ρ * total_energy(e_kin, e_pot, ts)
end
