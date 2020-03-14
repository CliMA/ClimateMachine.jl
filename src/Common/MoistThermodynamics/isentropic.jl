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
air_pressure_given_θ(
    θ::FT,
    Φ::FT,
    ::DryAdiabaticProcess,
    param_set::APS{FT} = MTPS{FT}(),
) where {FT} =
    MSLP(param_set) *
    (1 - Φ / (θ * cp_d(param_set)))^(cp_d(param_set) / R_d(param_set))

"""
    air_pressure(T::FT, T∞::FT, p∞::FT, ::DryAdiabaticProcess)

The air pressure for an isentropic process, where

 - `T` temperature
 - `T∞` ambient temperature
 - `p∞` ambient pressure
"""
air_pressure(
    T::FT,
    T∞::FT,
    p∞::FT,
    ::DryAdiabaticProcess,
    param_set::APS{FT} = MTPS{FT}(),
) where {FT} = p∞ * (T / T∞)^(FT(1) / kappa_d(param_set))

"""
    air_temperature(p::FT, θ::FT, Φ::FT, ::DryAdiabaticProcess)

The air temperature for an isentropic process, where

 - `p` pressure
 - `θ` potential temperature
"""
air_temperature(
    p::FT,
    θ::FT,
    ::DryAdiabaticProcess,
    param_set::APS{FT} = MTPS{FT}(),
) where {FT} = (p / MSLP(param_set))^(R_d(param_set) / cp_d(param_set)) * θ
