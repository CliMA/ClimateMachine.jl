module SubgridScaleParameters

using CLIMA.ParametersType
  
  @exportparameter C_smag         0.15                    "Standard Smagorinsky Coefficient"
  @exportparameter inv_Pr_turb    3                       "Turbulent Prandtl Number" 
  @exportparameter Prandtl_air    71//100                 "Molecular Prandtl Number, dry air" 
  @exportparameter c_a_KASM       0.10                    "cₐ KASM (2006)"
  @exportparameter c_e1_KASM      0.19                    "cₑ₁ KASM (2006)"
  @exportparameter c_e2_KASM      0.51                    "cₑ₂ KASM (2006)"
  @exportparameter c_1_KASM       c_a_KASM*0.76^2         "c₁  KASM (2006)"
  @exportparameter c_2_KASM       c_e2_KASM+2*c_1_KASM    "c₂ KASM (2006)"
  @exportparameter c_3_KASM       c_a_KASM^(3/2)          "c₃ KASM (2006)"

  end
