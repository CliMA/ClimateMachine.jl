CLIMA.Parameters.Microphysics.MP_n_0(ps::ParameterSet{FT}) where {FT} =
    FT(8e6 * 2)
CLIMA.Parameters.Microphysics.C_drag(ps::ParameterSet{FT}) where {FT} = FT(0.55)
CLIMA.Parameters.Microphysics.τ_cond_evap(ps::ParameterSet{FT}) where {FT} =
    FT(10)
CLIMA.Parameters.Microphysics.q_liq_threshold(ps::ParameterSet{FT}) where {FT} =
    FT(5e-4)
CLIMA.Parameters.Microphysics.τ_acnv(ps::ParameterSet{FT}) where {FT} = FT(1e3)
CLIMA.Parameters.Microphysics.E_col(ps::ParameterSet{FT}) where {FT} = FT(0.8)
CLIMA.Parameters.Microphysics.a_vent(ps::ParameterSet{FT}) where {FT} = FT(1.5)
CLIMA.Parameters.Microphysics.b_vent(ps::ParameterSet{FT}) where {FT} = FT(0.53)
CLIMA.Parameters.Microphysics.K_therm(ps::ParameterSet{FT}) where {FT} =
    FT(2.4e-2)
CLIMA.Parameters.Microphysics.D_vapor(ps::ParameterSet{FT}) where {FT} =
    FT(2.26e-5)
CLIMA.Parameters.Microphysics.ν_air(ps::ParameterSet{FT}) where {FT} =
    FT(1.6e-5)
CLIMA.Parameters.Microphysics.N_Sc(ps::ParameterSet{FT}) where {FT} =
    FT(ν_air / D_vapor)
