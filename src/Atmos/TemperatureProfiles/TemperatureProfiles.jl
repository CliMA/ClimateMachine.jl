module TemperatureProfiles

using DocStringExtensions

export TemperatureProfile,
    IsothermalProfile, DecayingTemperatureProfile, DryAdiabaticProfile

using CLIMAParameters: AbstractParameterSet
using CLIMAParameters.Planet: R_d, MSLP, cp_d, grav, T_surf_ref, T_min_ref

"""
    TemperatureProfile

Specifies the temperature or virtual temperature profile for a reference state.

Instances of this type are required to be callable objects with the following signature

    T,p = (::TemperatureProfile)(param_set::AbstractParameterSet, z::FT) where {FT}

where `T` is the temperature or virtual temperature (in K), and `p` is the pressure (in Pa).
"""
abstract type TemperatureProfile{FT} end

"""
    IsothermalProfile(param_set, T_virt)
    IsothermalProfile(param_set, ::Type{FT<:AbstractFloat})

A uniform virtual temperature profile, which is implemented
as a special case of [`DecayingTemperatureProfile`](@ref).
"""
IsothermalProfile(param_set::AbstractParameterSet, T_virt::FT) where {FT} =
    DecayingTemperatureProfile{FT}(param_set, T_virt, T_virt)

function IsothermalProfile(
    param_set::AbstractParameterSet,
    ::Type{FT},
) where {FT}
    T_virt = FT(T_surf_ref(param_set))
    return DecayingTemperatureProfile{FT}(param_set, T_virt, T_virt)
end

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
    ) where {FT}
        return new{FT}(T_surface, _T_min_ref)
    end
end

"""
    (profile::DryAdiabaticProfile)(
        param_set::AbstractParameterSet,
        z::FT,
    ) where {FT}

Returns dry adiabatic temperature and pressure profiles
with zero relative humidity. The temperature is truncated
to be greater than or equal to `profile.T_min_ref`.
"""
function (profile::DryAdiabaticProfile)(
    param_set::AbstractParameterSet,
    z::FT,
) where {FT}

    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

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
        H_t::FT = FT(R_d(param_set)) * _T_virt_surf / FT(grav(param_set)),
    ) where {FT}
        return new{FT}(_T_virt_surf, _T_min_ref, H_t)
    end
end


function (profile::DecayingTemperatureProfile)(
    param_set::AbstractParameterSet,
    z::FT,
) where {FT}
    _R_d::FT = R_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    # Scale height for surface temperature
    H_sfc = _R_d * profile.T_virt_surf / _grav
    H_t = profile.H_t
    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = profile.T_virt_surf - profile.T_min_ref
    Tv = profile.T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / profile.T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = _MSLP * exp(p)
    return (Tv, p)
end

end
