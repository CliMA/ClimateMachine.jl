### Reference state
using DocStringExtensions
export NoReferenceState,
    HydrostaticState,
    IsothermalProfile,
    DecayingTemperatureProfile,
    DryAdiabaticProfile

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

"""
    ReferenceState

Reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

"""
    TemperatureProfile

Specifies the temperature or virtual temperature profile for a reference state.

Instances of this type are required to be callable objects with the following signature

    T,p = (::TemperatureProfile)(orientation::Orientation, aux::Vars)

where `T` is the temperature or virtual temperature (in K), and `p` is the pressure (in Pa).
"""
abstract type TemperatureProfile{FT <: AbstractFloat} end

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
    T_virt, p =
        m.virtual_temperature_profile(atmos.orientation, atmos.param_set, aux)
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
    IsothermalProfile(param_set, T_virt)

A uniform virtual temperature profile, which is implemented
as a special case of [`DecayingTemperatureProfile`](@ref).
"""
IsothermalProfile(param_set::AbstractParameterSet, T_virt::FT) where {FT} =
    DecayingTemperatureProfile{FT}(param_set, T_virt, T_virt)

"""
    DryAdiabaticProfile{FT} <: TemperatureProfile{FT}


A temperature profile that has uniform dry potential temperature `θ`

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DryAdiabaticProfile{FT} <: TemperatureProfile{FT}
    "Surface temperature (K)"
    T_surface::FT
    "Minimum temperature (K)"
    T_min_ref::FT
    function DryAdiabaticProfile{FT}(
        param_set::AbstractParameterSet,
        T_surface::FT = FT(T_surf_ref(param_set)),
        _T_min_ref::FT = FT(T_min_ref(param_set)),
    ) where {FT <: AbstractFloat}
        return new{FT}(T_surface, _T_min_ref)
    end
end

"""
    (profile::DryAdiabaticProfile)(
        orientation::Orientation,
        param_set::AbstractParameterSet,
        aux::Vars,
    )

Returns dry adiabatic temperature and pressure profiles
with zero relative humidity. The temperature is truncated
to be greater than or equal to `profile.T_min_ref`.
"""
function (profile::DryAdiabaticProfile)(
    orientation::Orientation,
    param_set::AbstractParameterSet,
    aux::Vars,
)
    FT = eltype(aux)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    z = altitude(orientation, param_set, aux)

    # Temperature
    Γ = _grav / _cp_d
    T = max(profile.T_surface - Γ * z, profile.T_min_ref)

    # Pressure
    p = _MSLP * (T / profile.T_surface)^(_grav / (_R_d * Γ))
    if T == profile.T_min_ref
        z_top = (profile.T_surface - profile.T_min_ref) / Γ
        H_min = _R_d * profile.T_min_ref / _grav
        p *= exp(-(z - z_top) / H_min)
    end
    return (T, p)
end

"""
    DecayingTemperatureProfile{F} <: TemperatureProfile{FT}

A virtual temperature profile that decays smoothly with height `z`, from
`T_virt_surf` to `T_min_ref` over a height scale `H_t`. The default height
scale `H_t` is the density scale height evaluated with `T_virt_surf`.

```math
T_{\\text{v}}(z) = \\max(T_{\\text{v, sfc}} − (T_{\\text{v, sfc}} - T_{\\text{v, min}}) \\tanh(z/H_{\\text{t}})
```

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DecayingTemperatureProfile{FT} <: TemperatureProfile{FT}
    "Virtual temperature at surface (K)"
    T_virt_surf::FT
    "Minimum virtual temperature at the top of the atmosphere (K)"
    T_min_ref::FT
    "Height scale over which virtual temperature drops (m)"
    H_t::FT
    function DecayingTemperatureProfile{FT}(
        param_set::AbstractParameterSet,
        _T_virt_surf::FT = FT(T_surf_ref(param_set)),
        _T_min_ref::FT = FT(T_min_ref(param_set)),
        H_t::FT = FT(R_d(param_set)) * FT(T_surf_ref(param_set)) /
                  FT(grav(param_set)),
    ) where {FT <: AbstractFloat}
        return new{FT}(_T_virt_surf, _T_min_ref, H_t)
    end
end


function (profile::DecayingTemperatureProfile)(
    orientation::Orientation,
    param_set::AbstractParameterSet,
    aux::Vars,
)
    z = altitude(orientation, param_set, aux)
    ΔTv = profile.T_virt_surf - profile.T_min_ref
    Tv = profile.T_virt_surf - ΔTv * tanh(z / profile.H_t)
    FT = typeof(z)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    ΔTv′ = ΔTv / profile.T_virt_surf
    p = -z - profile.H_t * ΔTv′ * log(cosh(z / profile.H_t) - atanh(ΔTv′))
    p /= profile.H_t * (1 - ΔTv′^2)
    p = _MSLP * exp(p)
    return (Tv, p)
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
