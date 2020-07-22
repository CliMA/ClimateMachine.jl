### Soil heat model
export SoilHeatModel

abstract type AbstractHeatModel <: BalanceLaw end

"""
    SoilHeatModel <: BalanceLaw

The balance law for internal energy in soil.

"""
struct SoilHeatModel <: AbstractHeatModel end

"""
    vars_state(soil::AbstractHeatModel, ::Prognostic, FT)

"""
function vars_state(soil::AbstractHeatModel, ::Prognostic, FT)
    @vars()
end

"""
    ConstantInternalEnergy{FT} <: AbstractHeatModel
"""
struct ConstantInternalEnergy{FT} <: AbstractHeatModel
    T::FT
end

"""
    get_temperature(m::ConstantInternalEnergy)

"""
get_temperature(m::ConstantInternalEnergy) = m.T
