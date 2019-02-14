"""
    SurfaceFluxes

Surface flux functions, e.g., for buoyancy flux,
friction velocity, and exchange coefficients.

References:

Ref. Byun1990:
  Byun, Daewon W. "On the analytical solutions of flux-profile
  relationships for the atmospheric surface layer." Journal of
  Applied Meteorology 29.7 (1990): 652-657.
  http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2

Ref. Wyngaard1975:
  Wyngaard, John C. "Modeling the planetary boundary layer-Extension
  to the stable case." Boundary-Layer Meteorology 9.4 (1975): 441-460.
  https://link.springer.com/article/10.1007/BF00223393

Ref. Nishizawa2018:
  Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the
  Monin-Obukhov Similarity for Finite Volume Models." Journal of
  Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.
  https://doi.org/10.1029/2018MS001534

"""
module SurfaceFluxes

using Utilities.RootSolvers
using Utilities.MoistThermodynamics
using PlanetParameters

# ******************************************************************* Ref. Byun1990

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

""" Computes ψ_m for stable case. See Eq. 12 Ref. Byun1990 """
ψ_m_stable(ζ::R, ζ_0::R, β_m::R) where R = -β_m * (ζ - ζ_0)

""" Computes ψ_h for stable case. See Eq. 13 Ref. Byun1990 """
ψ_h_stable(ζ::R, ζ_0::R, β_h::R) where R = -β_h * (ζ - ζ_0)

"""
Computes roots of friction velocity equation (Eq. 10 in Ref. Byun1990)

  windspeed = u_* ( ln(z/z_0) - ψ_m(z/Λ, z_0/Λ) ) /κ        Eq. 10 in Ref. Byun1990
  Λ = -u_*^3/(buoyancy_flux κ)                              Eq. 22 in Ref. Wyngaard1975
  ψ_m(ζ, ζ_0) = [ψ_m_stable, ψ_m_unstable]                  Eq. 12, 14 in Ref. Byun1990

"""
function compute_friction_velocity(windspeed::R,
                                   buoyancy_flux::R,
                                   z_0::R,
                                   z_1::R,
                                   β_m::R,
                                   γ_m::R,
                                   tol_abs::R,
                                   iter_max::Int
                                   )::R where R
  # use neutral condition as first guess
  logz = log(z_1 / z_0)
  κ = PlanetParameters.k_Karman
  ustar_0 = windspeed * κ / logz
  ustar = ustar_0

  compute_Λ_MO(u) = - u^3 / (buoyancy_flux * κ)
  fixed_cond = z_1/compute_Λ_MO(ustar_0) >= 0
  fixed_compute_ψ_m(ζ, ζ_0) = fixed_cond ? ψ_m_stable(ζ, ζ_0, β_m) : ψ_m_unstable(ζ, ζ_0, γ_m)

  function compute_ψ_m(u)
    Λ_MO = compute_Λ_MO(u)
    ζ   = z_1/Λ_MO
    ζ_0 = z_0/Λ_MO
    return fixed_compute_ψ_m(ζ, ζ_0)
  end

  compute_u_star_roots(u) = windspeed - u / κ * (logz - compute_ψ_m(u))
  compute_ustar(u) = windspeed * κ / (logz - compute_ψ_m(u))

  if (abs(buoyancy_flux) > 0)
    ustar_1 = compute_ustar(ustar_0)
    args = ()
    ustar, converged = RootSolvers.find_zero(compute_u_star_roots,
                                             ustar_0, ustar_1,
                                             args,
                                             IterParams(tol_abs, iter_max),
                                             SecantMethod()
                                             )
  end
  return ustar
end

"""
Computes exchange transfer coefficients:

  C_D  momentum exchange coefficient
  C_H  thermodynamic exchange coefficient
  Λ_mo Monin-Obukhov length
"""
function exchange_coefficients_byun(Ri::R,
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

# ******************************************************************* Ref. Nishizawa2018

""" Computes Monin-Obukhov length. Eq. 3 Ref. Nishizawa2018 """
compute_Λ_MO(u, θ, flux) = - u^3* θ / (k_Karman * grav * flux)

""" Computes ψ_m. Eq. A1 Ref. Nishizawa2018 """
function compute_ψ_m(ζ)
  a = 4.7
  Pr_0 = 0.74
  Λ_MO = compute_Λ_MO(u)
  ζ   = z_1/Λ_MO
  ζ_0 = z_0/Λ_MO
  return fixed_compute_ψ_m(ζ, ζ_0)
end

function compute_friction_velocity(u, θ, flux, z, z_0)
  L = compute_Λ_MO(u, θ, flux)
  logz = log(z/z_0)
  u_star = k_Karman/(logz - compute_ψ_m(z_0/L) + ψ_m(z_0/L))
end






end