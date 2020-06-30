### Soil water model

export SoilWaterModel

"""
    SoilWaterModel{FT, IF, VF, MF, HM} <: BalanceLaw

The necessary components for Richard's Equation for water in soil.

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SoilWaterModel{FT, IF, VF, MF, HM} <: BalanceLaw
    "Impedance Factor - 1 or ice dependent"
    impedance_factor::IF
    "Viscosity Factor - 1 or temperature dependent"
    viscosity_factor::VF
    "Moisture Factor - 1 or moisture dependent"
    moisture_factor::MF
    "Hydraulics Model - used in matric potential and moisture factor of hydraulic conductivity."
    hydraulics::HM # vG, B/C, Haverkamp
end


#Defaults imply a constant hydraulic K = K sat
function SoilWaterModel(
    ::Type{FT};##we need to tell it what FT is in a non-optional way.
    impedance_factor::AbstractImpedanceFactor{FT} = NoImpedance{FT}(),
    viscosity_factor::AbstractViscosityFactor{FT} = ConstantViscosity{FT}(),
    moisture_factor::AbstractMoistureFactor{FT} = MoistureIndependent{FT}(),
    hydraulics::AbstractHydraulicsModel{FT} = vanGenuchten{FT}(),
) where {FT}
    args = (impedance_factor, viscosity_factor, moisture_factor, hydraulics)
    return SoilWaterModel{FT, typeof.(args)...}(args...)
end
