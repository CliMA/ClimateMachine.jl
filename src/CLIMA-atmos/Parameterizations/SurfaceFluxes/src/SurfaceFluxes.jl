"""
    SurfaceFluxes

Surface flux functions, e.g., for buoyancy flux,
friction velocity, and exchange coefficients.

"""
module SurfaceFluxes

using Utilities.RootSolvers
using Utilities.MoistThermodynamics
using PlanetParameters

function compute_buoyancy_flux(
  shf::R,
  lhf::R,
  T_b::R,
  qt_b::R,
  ql_b::R,
  qi_b::R,
  alpha0_0::R
  )::R where R
  cp_ = cp_m(qt_b, ql_b, qi_b)
  lv = latent_heat_vapor(T_b)
  temp1 = (PlanetParameters.eps_vi-1.0)
  temp2 = (shf + temp1 * cp_ * T_b * lhf /lv)
  return (PlanetParameters.grav * alpha0_0 / cp_ / T_b * temp2)
end

function ψ_m_unstable(ζ::R, ζ_0::R) where R
  x(ζ) = (1.0 - PlanetParameters.gamma_m * ζ)^0.25
  temp(ζ, ζ_0) = log((1.0 + ζ)/(1.0 + ζ_0))
  ψ_m = (2.0 * temp(x(ζ), x(ζ_0)) + temp(x(ζ)^2, x(ζ_0)^2) - 2.0 * atan(x(ζ)) + 2.0 * atan(x(ζ_0)))
  return ψ_m
end

function ψ_h_unstable(ζ::R, ζ_0::R) where R
  y(ζ) = sqrt(1.0 - PlanetParameters.gamma_h * ζ )
  ψ_h = 2.0 * log((1.0 + y(ζ))/(1.0 + y(ζ_0)))
  return ψ_h
end
ψ_m_stable(ζ::R, ζ_0::R) where R = -PlanetParameters.beta_m * (ζ - ζ_0)
ψ_h_stable(ζ::R, ζ_0::R) where R = -PlanetParameters.beta_h * (ζ - ζ_0)

"""
Computes roots of equation (eq. 10 in Ref. 1)

  windspeed = u_* ( ln(z/z_0) - ψ_m(z/Λ, z_0/Λ) ) /κ        eq. 10 in Ref. 1
  Λ = -u_*^3/(buoyancy_flux κ)                              eq. 22 in Ref. 2
  ψ_m(ζ, ζ_0) = [ψ_m_stable, ψ_m_unstable]                  eq. 12, 14 in Ref 1

Ref 1: Byun, Daewon W. "On the analytical solutions of flux-profile
       relationships for the atmospheric surface layer." Journal of
       Applied Meteorology 29.7 (1990): 652-657.
       https://journals.ametsoc.org/doi/pdf/10.1175/1520-0450%281990%29029%3C0652%3AOTASOF%3E2.0.CO%3B2

Ref 2: Wyngaard, John C. "Modeling the planetary boundary layer-Extension
       to the stable case." Boundary-Layer Meteorology 9.4 (1975): 441-460.
       https://sci-hub.tw/https://link.springer.com/article/10.1007/BF00223393
"""
function compute_friction_velocity(windspeed::R,
                                   buoyancy_flux::R,
                                   z_0::R,
                                   z_1::R,
                                   tol_abs::R,
                                   iter_max::Int
                                   )::R where R
  # use neutral condition as first guess
  logz = log(z_1 / z_0)
  κ = PlanetParameters.kappa_vk
  ustar_0 = windspeed * κ / logz
  ustar = ustar_0

  compute_Λ_MO(u) = - u * u * u / (buoyancy_flux * κ)
  fixed_cond = z_1/compute_Λ_MO(ustar_0) >= 0.0
  fixed_compute_ψ_m(ζ, ζ_0) = fixed_cond ? ψ_m_stable(ζ, ζ_0) : ψ_m_unstable(ζ, ζ_0)

  function compute_ψ_m(u)
    Λ_MO = compute_Λ_MO(u)
    ζ   = z_1/Λ_MO
    ζ_0 = z_0/Λ_MO
    return fixed_compute_ψ_m(ζ, ζ_0)
  end

  compute_u_star_roots(u) = windspeed - u / κ * (logz - compute_ψ_m(u))
  compute_ustar(u) = windspeed * κ / (logz - compute_ψ_m(u))

  if (abs(buoyancy_flux) > 1.0e-20)
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
Ref 1: Monin-Obukhov similarity. Daewon W. Byun, 1990: On the
       Analytical Solutions of Flux-Profile Relationships for
       the Atmospheric Surface Layer. J. Appl. Meteor., 29, 652–657.
       http://dx.doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2
"""
function exchange_coefficients_byun(Ri::R, z_b::R, z_0::R)::Tuple{R,R,R} where R
  γ_m = PlanetParameters.gamma_m
  γ_h = PlanetParameters.gamma_h
  β_h = PlanetParameters.beta_h
  β_m = PlanetParameters.beta_m
  κ = PlanetParameters.kappa_vk
  logz = log(z_b/z_0)
  zfactor = z_b/(z_b-z_0)*logz
  sb = Ri/Pr_0
  if Ri > 0.0
    ζ = zfactor/(2.0*β_h*(β_m*Ri -1.0))*((1.0-2.0*β_h*Ri)-sqrt(1.0+4.0*(β_h - β_m)*sb))
    Λ_mo = z_b/ζ
    ζ_0 = z_0/Λ_mo
    ψ_m = ψ_m_stable(ζ, ζ_0)
    ψ_h = ψ_h_stable(ζ, ζ_0)
  else
    qb = 1.0/9.0 * (1.0 /(γ_m * γ_m) + 3.0 * γ_h/γ_m * sb * sb)
    pb = 1.0/54.0 * (-2.0/(γ_m*γ_m*γ_m) + 9.0/γ_m * (-γ_h/γ_m + 3.0)*sb * sb)
    crit = qb * qb *qb - pb * pb
    if crit < 0.0
      tb = cbrt(sqrt(-crit) + fabs(pb))
      ζ = zfactor * (1.0/(3.0*γ_m)-(tb + qb/tb))
    else
      angle_ = acos(pb/sqrt(qb * qb * qb))
      ζ = zfactor * (-2.0 * sqrt(qb) * cos(angle_/3.0)+1.0/(3.0*γ_m))
    end
    Λ_mo = z_b/ζ
    ζ_0 = z_0/Λ_mo
    ψ_m = ψ_m_unstable(ζ, ζ_0)
    ψ_h = ψ_h_unstable(ζ, ζ_0)
  end
  cu = κ/(logz-ψ_m)
  cth = κ/(logz-ψ_h)/Pr_0
  cm = cu * cu
  ch = cu * cth
  return cm, ch, Λ_mo
end

end