"""
    MicrophysicsParameters

Module containing 1-moment bulk microphysics parameters.
"""
module MicrophysicsParameters
using CLIMA.ParametersType

@exportparameter MP_n_0          8e6 * 2         "Marshall-Palmer distribution n_0 coeff [1/m4]"
@exportparameter C_drag          0.55            "drag coefficient for rain drops [-]"

@exportparameter τ_cond_evap     10              "condensation/evaporation timescale [s]"

@exportparameter q_liq_threshold 5e-4            "autoconversion threshold [-]  ∈(0.5, 1) * 1e-3 "
@exportparameter τ_acnv          1e3             "autoconversion timescale [s]  ∈(1e3, 1e4) "

@exportparameter E_col           0.8             "collision efficiency [-]"

@exportparameter a_vent          1.5             "ventilation factor coefficient [-]"
@exportparameter b_vent          0.53            "ventilation factor coefficient [-]"
@exportparameter K_therm         2.4e-2          "thermal conductivity of air [J/m/s/K] "
@exportparameter D_vapor         2.26e-5         "diffusivity of water vapor [m2/s]"
@exportparameter ν_air           1.6e-5          "kinematic viscosity of air [m2/s]"
@exportparameter N_Sc            ν_air/D_vapor   "Schmidt number [-]"
end
