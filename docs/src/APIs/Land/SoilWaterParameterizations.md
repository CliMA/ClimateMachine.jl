# Soil Water Parameterizations

```@meta
CurrentModule = ClimateMachine.Land.SoilWaterParameterizations
```

## Water functions
```@docs
AbstractImpedanceFactor
NoImpedance
IceImpedance
impedance_factor
AbstractViscosityFactor
ConstantViscosity    
TemperatureDependentViscosity
viscosity_factor
AbstractMoistureFactor
MoistureDependent
MoistureIndependent
moisture_factor
AbstractHydraulicsModel
vanGenuchten
BrooksCorey
Haverkamp
hydraulic_conductivity
effective_saturation
pressure_head
hydraulic_head
matric_potential
volumetric_liquid_fraction
```