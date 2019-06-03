"""
    MicrophysicsParameters

Module containing 1-moment bulk microphysics parameters.
"""
module MicrophysicsParameters
using ..ParametersType

@exportparameter MP_n_0          TODO "Marshal-Palmer distribution n_0 coeff"
@exportparameter C_drag          TODO "drag coefficient for rain drops"

@exportparameter τ_cond_evap     1 "condensation/evaporation timescale [s]"
@exportparameter τ_subl_resubl   1 "sublimation/resublimation timescale [s]"

@exportparameter q_liq_threshold 5e-4  "autoconversion threshold [kg/kg]  ∈(0.5, 1) * 1e-3 "
@exportparameter τ_acnv          1e3   "autoconversion timescale [s]      ∈(1e3, 1e4) "

#@exportparameter τ_accr          TODO  "accretion timescale [s]"
@exportparameter E_col           TODO  "collision efficiency"

