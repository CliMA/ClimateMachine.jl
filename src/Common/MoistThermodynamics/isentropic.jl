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
air_pressure_given_θ(param_set::AbstractParameterSet{FT}, ::DryAdiabaticProcess, θ::FT, Φ::FT) where {FT} =
  FT(MSLP) * (1 - Φ / (θ * FT(cp_d))) ^ (FT(cp_d) / FT(R_d))
air_pressure_given_θ(args...) = air_pressure_given_θ(MoistThermoDefaultParameterSet{eltype(last(args))}(), args...)

"""
    air_pressure(T::FT, T∞::FT, p∞::FT, ::DryAdiabaticProcess)

The air pressure for an isentropic process, where

 - `T` temperature
 - `T∞` ambient temperature
 - `p∞` ambient pressure
"""
air_pressure(param_set::AbstractParameterSet{FT}, ::DryAdiabaticProcess, T::FT, T∞::FT, p∞::FT) where {FT} =
  p∞ * (T / T∞) ^ (FT(1) / FT(kappa_d))

"""
    air_temperature(p::FT, θ::FT, Φ::FT, ::DryAdiabaticProcess)

The air temperature for an isentropic process, where

 - `p` pressure
 - `θ` potential temperature
"""
air_temperature(param_set::AbstractParameterSet{FT}, ::DryAdiabaticProcess, p::FT, θ::FT) where {FT} =
  (p / FT(MSLP)) ^ (FT(R_d) / FT(cp_d)) * θ

air_temperature(dap::DryAdiabaticProcess, p::FT, θ::FT) where {FT<:Real} =
  air_temperature(MoistThermoDefaultParameterSet{FT}(), dap, p, θ)
