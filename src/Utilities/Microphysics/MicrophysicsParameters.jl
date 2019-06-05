"""
    MicrophysicsParameters

Module containing 1-moment bulk microphysics parameters.
"""
module MicrophysicsParameters
using ..ParametersType

@exportparameter MP_n_0          8e6     "Marshal-Palmer distribution n_0 coeff [1/m4]"
@exportparameter C_drag          0.55    "drag coefficient for rain drops [-]"

@exportparameter τ_cond_evap     1       "condensation/evaporation timescale [s]"

@exportparameter q_liq_threshold 5e-4    "autoconversion threshold [kg/kg]  ∈(0.5, 1) * 1e-3 "
@exportparameter τ_acnv          1e3     "autoconversion timescale [s]      ∈(1e3, 1e4) "

@exportparameter E_col           0.8     "collision efficiency"

end
