### Soil heat model
export PrescribedTemperatureModel

abstract type AbstractHeatModel <: AbstractSoilComponentModel end

"""
    PrescribedTemperatureModel{FT, F1} <: AbstractHeatModel

A model where the temperature is set by the user and not dynamically determined.

This is useful for driving Richard's equation without a back reaction on temperature.
# Fields
$(DocStringExtensions.FIELDS)
"""
struct PrescribedTemperatureModel{FT, F} <: AbstractHeatModel
    "Prescribed function for temperature"
    T::F
end

"""
    PrescribedTemperatureModel(
        ::Type{FT};
        T = (aux,t) -> eltype(aux)(288.0),
    ) where {FT}

Outer constructor for the PrescribedTemperatureModel defining default values, and
making it so changes to those defaults are supplied via keyword args.
"""
function PrescribedTemperatureModel(
    ::Type{FT};
    T = (aux, t) -> eltype(aux)(288.0),
) where {FT}
    return PrescribedTemperatureModel{FT, typeof(T)}(T)
end

"""
    function get_temperature(
        m::PrescribedTemperatureModel,
        aux::Vars,
        t::Real,
        state::Vars
    )    

Returns the temperature from the prescribed model.
"""
function get_temperature(
    m::PrescribedTemperatureModel,
    aux::Vars,
    t::Real,
    state::Vars
)
    return m.T(aux, t)
end
"""
    function get_initial_temperature(
        m::PrescribedTemperatureModel,
        aux::Vars,
        t::Real
    )    

Returns the temperature from the prescribed model.
Needed for soil_init_aux! of SoilWaterModel.
"""
function get_initial_temperature(
    m::PrescribedTemperatureModel,
    aux::Vars,
    t::Real,
)
    return m.T(aux, t)
end
