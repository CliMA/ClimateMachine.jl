abstract type TemperatureProfile{FT} end

struct DecayingTemperatureProfile{FT} <: TemperatureProfile{FT}
    "Virtual temperature at surface (K)"
    T_virt_surf::FT
    "Minimum virtual temperature at the top of the atmosphere (K)"
    T_min_ref::FT
    "Height scale over which virtual temperature drops (m)"
    H_t::FT
    function DecayingTemperatureProfile{FT}(
        params::NamedTuple,
        _T_virt_surf::FT,
        _T_min_ref::FT,
        H_t::FT = FT(params.R_d) * _T_virt_surf / FT(params.g),
    ) where {FT}
        return new{FT}(_T_virt_surf, _T_min_ref, H_t)
    end
end

function (profile::DecayingTemperatureProfile)(params::NamedTuple, z::FT) where {FT}
    R_d = params.R_d
    g   = params.g
    p₀  = params.pₒ

    # Scale height for surface temperature
    H_sfc = R_d * profile.T_virt_surf / g
    H_t = profile.H_t
    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv = profile.T_virt_surf - profile.T_min_ref
    Tv  = profile.T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / profile.T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = p₀ * exp(p)
    return (Tv, p)
end