### Reference state
using DocStringExtensions
using ..TemperatureProfiles
export NoReferenceState, HydrostaticState

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

"""
    ReferenceState

Reference state, for example, used as initial
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

A hydrostatic state specified by a virtual temperature profile and relative humidity.
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

    ρ = p / (_R_d * T_virt)
    aux.ref_state.ρ = ρ
    aux.ref_state.p = p
    # We evaluate the saturation vapor pressure, approximating
    # temperature by virtual temperature
    q_vap_sat = q_vap_saturation(atmos.param_set, T_virt, ρ)

    ρq_tot = ρ * relative_humidity(m) * q_vap_sat
    aux.ref_state.ρq_tot = ρq_tot

    q_pt = PhasePartition(ρq_tot)
    R_m = gas_constant_air(atmos.param_set, q_pt)
    T = T_virt * R_m / _R_d
    aux.ref_state.T = T
    aux.ref_state.ρe = ρ * internal_energy(atmos.param_set, T, q_pt)

    e_kin = F(0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    aux.ref_state.ρe = ρ * total_energy(atmos.param_set, e_kin, e_pot, T, q_pt)
end

"""
    relative_humidity(hs::HydrostaticState{P,FT})

Here, we enforce that relative humidity is zero
for a dry adiabatic profile.
"""
relative_humidity(hs::HydrostaticState{P, FT}) where {P, FT} =
    hs.relative_humidity
relative_humidity(hs::HydrostaticState{DryAdiabaticProfile, FT}) where {FT} =
    FT(0)
