module Microphysics

export MP_n_0,
    C_drag,
    τ_cond_evap,
    q_liq_threshold,
    τ_acnv,
    E_col,
    a_vent,
    b_vent,
    K_therm,
    D_vapor,
    ν_air,
    N_Sc

""" Marshall-Palmer distribution n_0 coeff [1/m4] """
function MP_n_0 end
""" drag coefficient for rain drops [-] """
function C_drag end

""" condensation/evaporation timescale [s] """
function τ_cond_evap end

""" autoconversion threshold [-]  ∈(0.5, 1) * 1e-3  """
function q_liq_threshold end
""" autoconversion timescale [s]  ∈(1e3, 1e4)  """
function τ_acnv end

""" collision efficiency [-] """
function E_col end

""" ventilation factor coefficient [-] """
function a_vent end
""" ventilation factor coefficient [-] """
function b_vent end
""" thermal conductivity of air [J/m/s/K]  """
function K_therm end
""" diffusivity of water vapor [m2/s] """
function D_vapor end
""" kinematic viscosity of air [m2/s] """
function ν_air end
""" Schmidt number [-] """
function N_Sc end

end # module Microphysics
