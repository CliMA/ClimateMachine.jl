"""
  Parameters for subgrid-scale turbuence module
"""
module SubgridScaleParameters

using CLIMA.ParametersType

@exportparameter C_smag         0.23           "Standard Smagorinsky Coefficient"
@exportparameter Prandtl_turb   1//3           "Turbulent Prandtl Number" 
@exportparameter Prandtl_air    71//100        "Molecular Prandtl Number, air" 

end
