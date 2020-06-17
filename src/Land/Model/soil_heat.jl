### Soil heat model
export SoilHeatModel

abstract type AbstractHeatModel <: BalanceLaw end

"""
    SoilHeatModel <: BalanceLaw

The balance law for internal energy in soil.

"""
struct SoilHeatModel <: AbstractHeatModel end

"""
    vars_state_conservative(soil::AbstractHeatModel, FT)

"""
function vars_state_conservative(soil::AbstractHeatModel, FT)
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
