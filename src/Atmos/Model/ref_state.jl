### Reference state
using DocStringExtensions
export ReferenceState,
    NoReferenceState,
    HydrostaticState,
    IsothermalProfile,
    LinearTemperatureProfile,
    DecayingTemperatureProfile,
    DryAdiabaticProfile

using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav

"""
    ReferenceState

Reference state, for example, used as initial
condition or for linearization.
"""
abstract type ReferenceState end

vars_state(m::ReferenceState, FT) = @vars()
vars_gradient(m::ReferenceState, FT) = @vars()
vars_diffusive(m::ReferenceState, FT) = @vars()
vars_aux(m::ReferenceState, FT) = @vars()
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

A hydrostatic state specified by a temperature profile and relative humidity.
"""
struct HydrostaticState{P, F} <: ReferenceState
    temperatureprofile::P
    relativehumidity::F
end

vars_aux(m::HydrostaticState, FT) =
    @vars(ρ::FT, p::FT, T::FT, ρe::FT, ρq_tot::FT)


function atmos_init_aux!(
    m::HydrostaticState{P, F},
    atmos::AtmosModel,
    aux::Vars,
    geom::LocalGeometry,
) where {P, F}
    T, p = m.temperatureprofile(atmos.orientation, atmos.param_set, aux)
    FT = eltype(aux)
    _R_d::FT = R_d(atmos.param_set)

    aux.ref_state.T = T
    aux.ref_state.p = p
    aux.ref_state.ρ = ρ = p / (_R_d * T)
    q_vap_sat = q_vap_saturation(atmos.param_set, T, ρ)
    aux.ref_state.ρq_tot = ρq_tot = ρ * m.relativehumidity * q_vap_sat

    q_pt = PhasePartition(ρq_tot)
    aux.ref_state.ρe = ρ * internal_energy(atmos.param_set, T, q_pt)

    e_kin = F(0)
    e_pot = gravitational_potential(atmos.orientation, aux)
    aux.ref_state.ρe = ρ * total_energy(atmos.param_set, e_kin, e_pot, T, q_pt)
end



"""
    TemperatureProfile

Specifies the temperature profile for a reference state.

Instances of this type are required to be callable objects with the following signature

    T,p = (::TemperatureProfile)(orientation::Orientation, aux::Vars)

where `T` is the temperature (in K), and `p` is the pressure (in hPa).
"""
abstract type TemperatureProfile end


"""
    IsothermalProfile{F} <: TemperatureProfile

A uniform temperature profile.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct IsothermalProfile{F} <: TemperatureProfile
    "temperature (K)"
    T::F
end

function (profile::IsothermalProfile)(
    orientation::Orientation,
    param_set,
    aux::Vars,
)
    FT = eltype(aux)
    _R_d::FT = R_d(param_set)
    _MSLP::FT = MSLP(param_set)
    p =
        _MSLP *
        exp(-gravitational_potential(orientation, aux) / (_R_d * profile.T))
    return (profile.T, p)
end

"""
    DryAdiabaticProfile{F} <: TemperatureProfile


A temperature profile that has uniform potential temperature θ until it
reaches the minimum specified temperature `T_min`

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DryAdiabaticProfile{F} <: TemperatureProfile
    "minimum temperature (K)"
    T_min::F
    "potential temperature (K)"
    θ::F
end

function (profile::DryAdiabaticProfile)(
    orientation::Orientation,
    param_set,
    aux::Vars,
)
    FT = eltype(aux)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    LinearTemperatureProfile(profile.T_min, profile.θ, FT(_grav / _cp_d))(
        orientation,
        param_set,
        aux,
    )
end

"""
    LinearTemperatureProfile{F} <: TemperatureProfile

A temperature profile which decays linearly with height `z`, until it reaches a minimum specified temperature.

```math
T(z) = \\max(T_{\\text{surface}} − Γ z, T_{\\text{min}})
```

# Fields

$(DocStringExtensions.FIELDS)
"""
struct LinearTemperatureProfile{FT} <: TemperatureProfile
    "minimum temperature (K)"
    T_min::FT
    "surface temperature (K)"
    T_surface::FT
    "lapse rate (K/m)"
    Γ::FT
end

function (profile::LinearTemperatureProfile)(
    orientation::Orientation,
    param_set::PS,
    aux::Vars,
) where {PS}

    FT = eltype(aux)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    z = altitude(orientation, param_set, aux)
    T = max(profile.T_surface - profile.Γ * z, profile.T_min)

    p = _MSLP * (T / profile.T_surface)^(_grav / (_R_d * profile.Γ))
    if T == profile.T_min
        z_top = (profile.T_surface - profile.T_min) / profile.Γ
        H_min = _R_d * profile.T_min / _grav
        p *= exp(-(z - z_top) / H_min)
    end
    return (T, p)
end

"""
    DecayingTemperatureProfile{F} <: TemperatureProfile

A virtual temperature profile that decays smoothly with height `z`, dropping by a specified temperature difference `ΔTv` over a height scale `H_t`.

```math
Tv(z) = \\max(Tv{\\text{surface}} − ΔTv \\tanh(z/H_{\\text{t}})
```

# Fields

$(DocStringExtensions.FIELDS)
"""
struct DecayingTemperatureProfile{FT} <: TemperatureProfile
    "virtual temperature at surface (K)"
    Tv_surface::FT
    "virtual temperature drop from surface to top of the atmosphere (K)"
    ΔTv::FT
    "height scale over which virtual temperature drops (m)"
    H_t::FT
end

function (profile::DecayingTemperatureProfile)(
    orientation::Orientation,
    param_set::AbstractParameterSet,
    aux::Vars,
)
    z = altitude(orientation, param_set, aux)
    Tv = profile.Tv_surface - profile.ΔTv * tanh(z / profile.H_t)
    FT = typeof(z)
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    ΔTv_p = profile.ΔTv / profile.Tv_surface
    H_surface = _R_d * profile.Tv_surface / _grav
    p = -z - profile.H_t * ΔTv_p * log(cosh(z / profile.H_t) - atanh(ΔTv_p))
    p /= H_surface * (1 - ΔTv_p^2)
    p = _MSLP * exp(p)
    return (Tv, p)
end
