function get_spatial_SST_idealized( atmos, aux, state, SST_min )
    FT = eltype(state)
    # following Thatcher and Jablonowski 2016
    #SST_middn = FT(271) # SST at the poles
    φ = latitude( atmos.orientation, aux)
    Δφ = FT(26) * π / FT(180) # latitudinal width of Gaussian function
    ΔSST = FT(29) # Eq-pole SST difference in K
    T_sfc = ΔSST * exp( - φ^2 / ( 2 * Δφ^2  ) ) + SST_min

    eps =  FT(0.622)
    ρ = state.ρ
    e_int = internal_energy( atmos.moisture, atmos.orientation, state, aux)
    T = air_temperature( atmos.param_set, e_int)
    p = air_pressure( atmos.param_set, T, ρ)
    #p = FT(100000)
    T_0 = FT(273.16) # triple point of water
    e_0 = FT(610.78) # sat water pressure at T_0
    L = FT(2.5e6) # latent heat of vaporization at T_0
    R_v = FT(461.5) # gas constant for water vapor
    q_sfc = eps/ p * e_0 * exp( - L / R_v * (FT(1) / T_sfc - FT(1) / T_0) )
    return T_sfc, q_sfc
  end
