"""
  Parameters for subgrid-scale turbuence module
"""
module SubgridScaleParameters

using CLIMA.ParametersType

@exportparameter C_smag         0.23           "Standard Smagorinsky Coefficient"
@exportparameter Prandtl_turb   1//3           "Turbulent Prandtl Number" 

end
