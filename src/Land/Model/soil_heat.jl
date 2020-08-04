### Soil heat model
export SoilHeatModel, PrescribedTemperatureModel

abstract type AbstractHeatModel <: AbstractSoilComponentModel end

"""
    PrescribedTemperatureModel{FT} <: AbstractHeatModel

A model where the temperature is set by the user and not dynamically determined.

This is useful for driving Richard's equation without a back reaction on temperature.
# Fields
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PrescribedTemperatureModel{FT} <: AbstractHeatModel
    "Prescribed function for temperature"
    T::FT = FT(0.0)
end

"""
    get_temperature(m::PrescribedTemperatureModel)
"""
get_temperature(m::PrescribedTemperatureModel) = m.T



"""
    SoilHeatModel <: AbstractHeatModel

The balance law for internal energy in soil.

To be used when the user wants to dynamically determine internal energy and temperature.

"""
struct SoilHeatModel <: AbstractHeatModel end
