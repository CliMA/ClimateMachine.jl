#####
##### Formulas assuming dry (unsaturated) adiabatic (i.e. isentropic) processes
#####

export DryAdiabaticProcess

export air_pressure_given_θ, air_pressure, air_temperature

"""
    DryAdiabaticProcess

For dispatching to isentropic formulas
"""
struct DryAdiabaticProcess end

"""
    air_pressure_given_θ(θ::FT, Φ::FT, ::DryAdiabaticProcess)

The air pressure for an isentropic process, where

 - `θ` potential temperature
 - `Φ` gravitational potential
"""
function air_pressure_given_θ(
    param_set::APS,
    θ::FT,
    Φ::FT,
    ::DryAdiabaticProcess,
) where {FT<:AbstractFloat}
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    return _MSLP * (1 - Φ / (θ * _cp_d))^(_cp_d / _R_d)
end

"""
    air_pressure(T::FT, T∞::FT, p∞::FT, ::DryAdiabaticProcess)

The air pressure for an isentropic process, where

 - `T` temperature
 - `T∞` ambient temperature
 - `p∞` ambient pressure
"""
function air_pressure(
    param_set::APS,
    T::FT,
    T∞::FT,
    p∞::FT,
    ::DryAdiabaticProcess,
) where {FT<:AbstractFloat}
    _kappa_d::FT = kappa_d(param_set)
    return p∞ * (T / T∞)^(FT(1) / _kappa_d)
end

"""
    air_temperature(p::FT, θ::FT, Φ::FT, ::DryAdiabaticProcess)

The air temperature for an isentropic process, where

 - `p` pressure
 - `θ` potential temperature
"""
function air_temperature(
    param_set::APS,
    p::FT,
    θ::FT,
    ::DryAdiabaticProcess,
) where {FT<:AbstractFloat}
    _R_d::FT = R_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _MSLP::FT = MSLP(param_set)
    return (p / _MSLP)^(_R_d / _cp_d) * θ
end
