"""
    SurfaceFluxes

  Surface flux functions, e.g., for buoyancy flux,
  friction velocity, and exchange coefficients.

## Sub-modules
  - module Byun1990
  - module Nishizawa2018

## Interface
  - Each sub-module has the following functions:
    - [`compute_MO_len`](@ref) computes the Monin-Obukhov length
    - [`compute_friction_velocity`](@ref) computes the friction velocity
    - [`compute_buoyancy_flux`](@ref) computes the buoyancy flux
    - [`compute_exchange_coefficients`](@ref) computes the exchange coefficients

## References

  - Ref. Byun1990:
    - Byun, Daewon W. "On the analytical solutions of flux-profile
      relationships for the atmospheric surface layer." Journal of
      Applied Meteorology 29.7 (1990): 652-657.
      http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2

  - Ref. Wyngaard1975:
    - Wyngaard, John C. "Modeling the planetary boundary layer-Extension
      to the stable case." Boundary-Layer Meteorology 9.4 (1975): 441-460.
      https://link.springer.com/article/10.1007/BF00223393

  - Ref. Nishizawa2018:
    - Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the
      Monin-Obukhov Similarity for Finite Volume Models." Journal of
      Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.
      https://doi.org/10.1029/2018MS001534

"""
module SurfaceFluxes

using Utilities.RootSolvers
using Utilities.MoistThermodynamics
using PlanetParameters

module Byun1990
  using Utilities.RootSolvers
  using Utilities.MoistThermodynamics
  using PlanetParameters

  """ Computes ψ_m for stable case. See Eq. 12 Ref. Byun1990 """
  ψ_m_stable(ζ::R, ζ_0::R, β_m::R) where R = -β_m * (ζ - ζ_0)

  """ Computes ψ_h for stable case. See Eq. 13 Ref. Byun1990 """
  ψ_h_stable(ζ::R, ζ_0::R, β_h::R) where R = -β_h * (ζ - ζ_0)

  """ Computes ψ_m for unstable case. See Eq. 14 Ref. Byun1990 """
  function ψ_m_unstable(ζ::R, ζ_0::R, γ_m::R) where R
    x(ζ) = (1 - γ_m * ζ)^(1/4)
    temp(ζ, ζ_0) = log((1 + ζ)/(1 + ζ_0))
    ψ_m = (2 * temp(x(ζ), x(ζ_0)) + temp(x(ζ)^2, x(ζ_0)^2) - 2 * atan(x(ζ)) + 2 * atan(x(ζ_0)))
    return ψ_m
  end

  """ Computes ψ_h for unstable case. See Eq. 15 Ref. Byun1990 """
  function ψ_h_unstable(ζ::R, ζ_0::R, γ_h::R) where R
    y(ζ) = sqrt(1 - γ_h * ζ )
    ψ_h = 2 * log((1 + y(ζ))/(1 + y(ζ_0)))
    return ψ_h
  end

  # ******************* Interface funcs:

  """ Computes the Monin-Obukhov length (Eq. 3 Ref. Byun1990) """
  compute_MO_len(u, flux) = - u^3 / (flux * PlanetParameters.k_Karman)

  """
  Computes roots of friction velocity equation (Eq. 10 in Ref. Byun1990)

    windspeed = u_* ( ln(z/z_0) - ψ_m(z/L, z_0/L) ) /κ        Eq. 10 in Ref. Byun1990
  """
  function compute_friction_velocity(windspeed::R,
                                     flux::R,
                                     z_0::R,
                                     z_1::R,
                                     β_m::T,
                                     γ_m::T,
                                     tol_abs::AbstractFloat,
                                     iter_max::Int
                                     )::R where R where T

    logz = log(z_1 / z_0)
    κ = PlanetParameters.k_Karman
    ustar_0 = windspeed * κ / logz
    ustar = ustar_0
    let windspeed=windspeed, flux=flux,
        z_0=z_0, z_1=z_1, β_m=β_m, γ_m=γ_m,
        κ=κ, ustar_0=ustar_0, logz=logz

      # use neutral condition as first guess
      stable = z_1/compute_MO_len(ustar_0, flux) >= 0
      function compute_ψ_m(u, flux)
        Λ_MO = compute_MO_len(u, flux)
        ζ   = z_1/Λ_MO
        ζ_0 = z_0/Λ_MO
        return stable ? ψ_m_stable(ζ, ζ_0, β_m) : ψ_m_unstable(ζ, ζ_0, γ_m)
      end
      function compute_ustar(u, flux)
        return windspeed * κ / (logz - compute_ψ_m(u, flux)) # Eq. 10 in Ref. Byun1990
      end
      function compute_u_star_roots(u, flux) # Eq. 10 in Ref. Byun1990
        return u - compute_ustar(u, flux)
      end
      if (abs(flux) > 0)
        ustar_1 = compute_ustar(ustar_0, flux)
        args = (flux,)
        ustar, converged = RootSolvers.find_zero(compute_u_star_roots,
                                                 ustar_0, ustar_1,
                                                 args,
                                                 IterParams(tol_abs, iter_max),
                                                 SecantMethod()
                                                 )
      end

    end


    return ustar
  end

  """ Computes buoyancy flux. See Eq. 14 Ref. Byun1990 """
  function compute_buoyancy_flux(shf::R,
                                 lhf::R,
                                 T_b::R,
                                 qt_b::R,
                                 ql_b::R,
                                 qi_b::R,
                                 alpha0_0::R
                                 )::R where R
    cp_ = cp_m(qt_b, ql_b, qi_b)
    lv = latent_heat_vapor(T_b)
    temp1 = (PlanetParameters.eps_vi-1)
    temp2 = (shf + temp1 * cp_ * T_b * lhf /lv)
    return (PlanetParameters.grav * alpha0_0 / cp_ / T_b * temp2)
  end

  """
  Computes exchange transfer coefficients:

    C_D  momentum exchange coefficient
    C_H  thermodynamic exchange coefficient
    Λ_mo Monin-Obukhov length
  """
  function compute_exchange_coefficients(Ri::R,
                                         z_b::R,
                                         z_0::R,
                                         γ_m::R,
                                         γ_h::R,
                                         β_m::R,
                                         β_h::R,
                                         Pr_0::R
                                         )::Tuple{R,R,R} where R
    logz = log(z_b/z_0)
    zfactor = z_b/(z_b-z_0)*logz
    s_b = Ri/Pr_0
    if Ri > 0
      ζ = zfactor/(2*β_h*(β_m*Ri - 1))*((1-2*β_h*Ri)-sqrt(1+4*(β_h - β_m)*s_b)) # Eq. 19 in Ref. Byun1990
      Λ_mo = z_b/ζ                                             # LHS of Eq. 3 in Byun1990
      ζ_0 = z_0/Λ_mo
      ψ_m = ψ_m_stable(ζ, ζ_0, β_m)
      ψ_h = ψ_h_stable(ζ, ζ_0, β_h)
    else
      Q_b = 1/9 * (1 /(γ_m^2) + 3 * γ_h/γ_m * s_b^2)           # Eq. 31 in Ref. Byun1990
      P_b = 1/54 * (-2/(γ_m^3) + 9/γ_m * (-γ_h/γ_m + 3)*s_b^2) # Eq. 32 in Ref. Byun1990
      crit = Q_b * Q_b *Q_b - P_b * P_b
      if crit < 0
        T_b = cbrt(sqrt(-crit) + abs(P_b))                     # Eq. 34 in Ref. Byun1990
        ζ = zfactor * (1/(3*γ_m)-(T_b + Q_b/T_b))              # Eq. 29 in Ref. Byun1990
      else
        θ_b = acos(P_b/sqrt(Q_b * Q_b * Q_b))                  # Eq. 33 in Ref. Byun1990
        ζ = zfactor * (-2 * sqrt(Q_b) * cos(θ_b/3)+1/(3*γ_m))  # Eq. 28 in Ref. Byun1990
      end
      Λ_mo = z_b/ζ                                             # LHS of Eq. 3 in Byun1990
      ζ_0 = z_0/Λ_mo
      ψ_m = ψ_m_unstable(ζ, ζ_0, γ_m)
      ψ_h = ψ_h_unstable(ζ, ζ_0, γ_h)
    end
    cu = PlanetParameters.k_Karman/(logz-ψ_m)                  # Eq. 10 in Ref. Byun1990, solved for u^*
    cth = PlanetParameters.k_Karman/(logz-ψ_h)/Pr_0            # Eq. 11 in Ref. Byun1990, solved for h^*
    C_D = cu^2                                                 # Eq. 36 in Byun1990
    C_H = cu*cth                                               # Eq. 37 in Byun1990
    return C_D, C_H, Λ_mo
  end
end

module Nishizawa2018
  using Utilities.RootSolvers
  using Utilities.MoistThermodynamics
  using PlanetParameters

  """ Computes R_z0 expression, defined after Eq. 15 Ref. Nishizawa2018 """
  compute_R_z0(z_0::R, Δz::R) where R = 1 - z_0/Δz

  """ Computes f_m in Eq. A7 Ref. Nishizawa2018 """
  compute_f_m(ζ::R) where R = (1-15*ζ)^(1/4)

  """ Computes f_h in Eq. A8 Ref. Nishizawa2018 """
  compute_f_h(ζ::R) where R = (1-9*ζ)^(1/2)

  """ Computes f in Eq. A21 Ref. Nishizawa2018 """
  compute_f(ζ::R) where R = (1-16*ζ)^(1/4)

  """ Computes ϕ_m in Eq. A1 Ref. Nishizawa2018 """
  compute_ϕ_m(ζ::R, L::R, a::R) where R = L>=0 ? a*ζ+1 : (1-15*ζ)^(-1/4)

  """ Computes ϕ_h in Eq. A2 Ref. Nishizawa2018 """
  compute_ϕ_h(ζ::R, L::R, a::R, Pr::R) where R = L>=0 ? a*ζ/Pr+1 : (1-9*ζ)^(-1/2)

  """ Computes ψ_m in Eq. A3 Ref. Nishizawa2018 """
  function compute_ψ_m(ζ::R, L::R, a::R) where R
    f_m = compute_f_m(ζ)
    return L>=0 ? -a*ζ : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2
  end

  """ Computes ψ_h in Eq. A4 Ref. Nishizawa2018 """
  compute_ψ_h(ζ::R, L::R, a::R, Pr::R) where R = L>=0 ? -a*ζ/Pr : 2*log((1+compute_f_h(ζ))/2)

  """ Computes Ψ_m in Eq. A13 Ref. Nishizawa2018 """
  compute_Ψ_m_low_ζ_lim(ζ::R, a::R) where R = ζ>=0 ? -a*ζ/2 : -15*ζ/8

  """ Computes Ψ_h in Eq. A14 Ref. Nishizawa2018 """
  compute_Ψ_h_low_ζ_lim(ζ::R, a::R, Pr::R) where R = ζ>=0 ? -a*ζ/(2*Pr) : -9*ζ/4

  """ Computes Ψ_m in Eq. A5 Ref. Nishizawa2018 """
  function compute_Ψ_m(ζ::R, L::R, a::R, tol::R) where R
    if ζ < tol
      return compute_Ψ_m_low_ζ_lim(ζ, a)
    else
      f_m = compute_f_m(ζ)
      f = compute_f(ζ)
      return L>=0 ? -a*ζ/2 : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2-1+(1-f^3)/(12*ζ)
    end
  end

  """ Computes Ψ_h in Eq. A6 Ref. Nishizawa2018 """
  function compute_Ψ_h(ζ::R, L::R, a::R, Pr::R, tol::R) where R
    if ζ < tol
      return compute_Ψ_h_low_ζ_lim(ζ, a, Pr)
    else
      f_h = compute_f_h(ζ)
      return L>=0 ? -a*ζ/(2*Pr) : 2*log((1+f_h)/2) + 2*(1-f_h)/(9*ζ)
    end
  end

  # """
  # Computes u_ave in Computes friction
  # velocity, in Eq. 12 in Ref. Nishizawa2018
  # u_ave = u_star/κ * ( ln(Δz/z_0) - Ψ_m(Δz/L) + z_0/Δz * Ψ_m(z_0/L) + R_z0 [ψ_m(z_0/L) - 1] )
  # """
  # function compute_u_ave(u_star, θ, flux, Δz, z_0)
  #   L = compute_MO_len(u_star, θ, flux)
  #   logz = log(Δz/z_0)
  #   R_z0 = compute_R_z0(z_0, Δz)
  #   Ψ_m_0 = compute_Ψ_m(z_0/L)
  #   Ψ_m = compute_Ψ_m(Δz/L)
  #   ψ_m = compute_ψ_m(z_0/L)
  #   return u_star/k_Karman * (logz - Ψ_m + z_0/Δz * Ψ_m_0 + R_z0 * (ψ_m - 1) )
  # end

  # """ Computes roots of friction velocity equation (Eq. 12 in Ref. Nishizawa2018) """
  # function compute_u_star_roots(u_ave, u_star, θ, flux, Δz, z_0)
  #   return u_ave - compute_u_ave(u_star, θ, flux, Δz, z_0)
  # end

  """ Computes Monin-Obukhov length. Eq. 3 Ref. Nishizawa2018 """
  compute_MO_len(u, θ, flux) = - u^3* θ / (PlanetParameters.k_Karman * PlanetParameters.grav * flux)

  """
  Computes friction velocity, in Eq. 12 in
  Ref. Nishizawa2018, by solving the
  non-linear equation:

    u_ave = u_star/κ * ( ln(Δz/z_0) - Ψ_m(Δz/L) + z_0/Δz * Ψ_m(z_0/L) + R_z0 [ψ_m(z_0/L) - 1] )
    where L is a non-linear function of u_star (see compute_MO_len).
  """
  function compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, tol)
    # L = compute_MO_len(u_ave, θ, flux)
    # R_z0 = compute_R_z0(z_0, Δz)
    # Ψ_m = compute_Ψ_m(ζ, L, a, tol)
    # ψ_m = compute_ϕ_m(ζ, L, a)
    # logz = log(Δz/z_0)
    # u_star = PlanetParameters.k_Karman/(logz - compute_ψ_m(z_0/L) + ψ_m(z_0/L))
    # let u_ave=u_ave, θ=θ, flux=flux, Δz=Δz, z_0=z_0
    #   ustar_0 = 1
    #   ustar_1 = 1

    #   function Ψ_m_closure(u)
    #     L = compute_MO_len(u, θ, flux)
    #     logz = log(Δz/z_0)
    #     Ψ_m = compute_Ψ_m(ζ, L, a, tol)
    #     compute_Ψ_m(ζ, L, a, tol)
    #   end

    #   ψ_m_closure(ζ) = compute_ψ_m(ζ, L, a)

    #   function compute_ustar(L)
    #     temp = log(Δz/z_0) - Ψ_m_closure(Δz/L) + z_0/Δz * Ψ_m_closure(z_0/L) + R_z0 [ψ_m_closure(z_0/L) - 1]
    #     return u_ave*k_Karman/()
    #   end
    #   function compute_u_star_roots(L)
    #     return u - compute_ustar(u)
    #   end
    #   args = (flux,)

    #   ustar, converged = RootSolvers.find_zero(compute_u_star_roots,
    #                                            ustar_0, ustar_1,
    #                                            args,
    #                                            IterParams(tol_abs, iter_max),
    #                                            SecantMethod()
    #                                            )
    # end
    # return u_star
  end

end




end