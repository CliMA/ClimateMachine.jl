"""
    SurfaceFluxes

  Surface flux functions, e.g., for buoyancy flux,
  friction velocity, and exchange coefficients.

## Sub-modules
  - module Byun1990
  - module Nishizawa2018

## Interface
  - [`compute_buoyancy_flux`](@ref) computes the buoyancy flux
  - In addition, each sub-module has the following functions:
    - [`compute_MO_len`](@ref) computes the Monin-Obukhov length
    - [`compute_friction_velocity`](@ref) computes the friction velocity
    - [`compute_exchange_coefficients`](@ref) computes the exchange coefficients

## References

  - Ref. Byun1990:
    - Byun, Daewon W. "On the analytical solutions of flux-profile
      relationships for the atmospheric surface layer." Journal of
      Applied Meteorology 29.7 (1990): 652-657.
      https://doi.org/10.1175/1520-0450(1990)029<0652:OTASOF>2.0.CO;2

  - Ref. Wyngaard1975:
    - Wyngaard, John C. "Modeling the planetary boundary layer-Extension
      to the stable case." Boundary-Layer Meteorology 9.4 (1975): 441-460.
      https://link.springer.com/article/10.1007/BF00223393

  - Ref. Nishizawa2018:
    - Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the
      Monin-Obukhov Similarity for Finite Volume Models." Journal of
      Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.
      https://doi.org/10.1029/2018MS001534

  - Ref. Businger1971:
    - Businger, Joost A., et al. "Flux-profile relationships in the
      atmospheric surface layer." Journal of the atmospheric Sciences
      28.2 (1971): 181-189.

"""
module SurfaceFluxes

using ..RootSolvers
using ..MoistThermodynamics
using ..PlanetParameters

# export compute_buoyancy_flux

"""
    compute_buoyancy_flux(shf, lhf, T_b, qt_b, ql_b, qi_b, alpha0_0)

Computes buoyancy flux given sensible heat flux `shf`,
latent heat flux `lhf`, surface boundary temperature `T_b`,
total specific humidity `qt_b`, liquid specific humidity `ql_b`,
ice specific humidity `qi_b` and specific `alpha0_0`.
"""
function compute_buoyancy_flux(shf, lhf, T_b, qt_b, ql_b, qi_b, alpha0_0)
  cp_ = cp_m(PhasePartition(qt_b, ql_b, qi_b))
  lv = latent_heat_vapor(T_b)
  temp1 = (molmass_ratio-1)
  temp2 = (shf + temp1 * cp_ * T_b * lhf /lv)
  return (grav * alpha0_0 / cp_ / T_b * temp2)
end

module Byun1990

using ...RootSolvers
using ...MoistThermodynamics
using ...PlanetParameters

""" Computes ψ_m for stable case. See Eq. 12 Ref. Byun1990 """
ψ_m_stable(ζ, ζ_0, β_m) = -β_m * (ζ - ζ_0)

""" Computes ψ_h for stable case. See Eq. 13 Ref. Byun1990 """
ψ_h_stable(ζ, ζ_0, β_h) = -β_h * (ζ - ζ_0)

""" Computes ψ_m for unstable case. See Eq. 14 Ref. Byun1990 """
function ψ_m_unstable(ζ, ζ_0, γ_m)
  x(ζ) = sqrt(sqrt(1 - γ_m * ζ))
  temp(ζ, ζ_0) = log1p((ζ - ζ_0)/(1 + ζ_0))  # log((1 + ζ)/(1 + ζ_0))
  ψ_m = (2 * temp(x(ζ), x(ζ_0)) + temp(x(ζ)^2, x(ζ_0)^2) - 2 * atan(x(ζ)) + 2 * atan(x(ζ_0)))
  return ψ_m
end

""" Computes ψ_h for unstable case. See Eq. 15 Ref. Byun1990 """
function ψ_h_unstable(ζ, ζ_0, γ_h)
  y(ζ) = sqrt(1 - γ_h * ζ )
  ψ_h = 2 * log1p((y(ζ) - y(ζ_0))/(1 + y(ζ_0))) # log((1 + y(ζ))/(1 + y(ζ_0)))
  return ψ_h
end

"""
    compute_MO_len(u, flux)

Computes the Monin-Obukhov length (Eq. 3 Ref. Byun1990)
"""
compute_MO_len(u, flux) = - u^3 / (flux * k_Karman)

"""
    compute_friction_velocity(u_ave, flux, z_0, z_1, β_m, γ_m, tol_abs, iter_max)

Computes roots of friction velocity equation (Eq. 10 in Ref. Byun1990)

`u_ave = u_* ( ln(z/z_0) - ψ_m(z/L, z_0/L) ) /κ        Eq. 10 in Ref. Byun1990`
"""
function compute_friction_velocity(u_ave, flux, z_0, z_1, β_m, γ_m, tol_abs, iter_max)

  ustar_0 = u_ave * k_Karman / log(z_1 / z_0)
  ustar = ustar_0
  let u_ave=u_ave, flux=flux, z_0=z_0, z_1=z_1, β_m=β_m, γ_m=γ_m

    # use neutral condition as first guess
    stable = z_1/compute_MO_len(ustar_0, flux) >= 0
    function compute_ψ_m(u)
      L_MO = compute_MO_len(u, flux)
      ζ   = z_1/L_MO
      ζ_0 = z_0/L_MO
      return stable ? ψ_m_stable(ζ, ζ_0, β_m) : ψ_m_unstable(ζ, ζ_0, γ_m)
    end
    function compute_u_ave_over_ustar(u)
      return (log(z_1 / z_0) - compute_ψ_m(u)) / k_Karman # Eq. 10 in Ref. Byun1990
    end
    compute_ustar(u) = u_ave/compute_u_ave_over_ustar(u)

    if (abs(flux) > 0)
      ustar_1 = compute_ustar(ustar_0)
      sol = RootSolvers.find_zero(
        u -> u_ave - u*compute_u_ave_over_ustar(u),
        ustar_0, ustar_1, SecantMethod(), CompactSolution(),
        tol_abs, iter_max)
      ustar = sol.root
    end

  end

  return ustar
end

"""
    compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)

Computes exchange transfer coefficients:
 - C_D  momentum exchange coefficient      (Eq. 36)
 - C_H  thermodynamic exchange coefficient (Eq. 37)
 - L_mo Monin-Obukhov length               (re-arranged Eq. 3)
"""
function compute_exchange_coefficients(Ri, z_b, z_0, γ_m, γ_h, β_m, β_h, Pr_0)
  logz = log(z_b/z_0)
  zfactor = z_b/(z_b-z_0)*logz
  s_b = Ri/Pr_0
  if Ri > 0
    temp = ((1-2*β_h*Ri)-sqrt(1+4*(β_h - β_m)*s_b))
    ζ = zfactor/(2*β_h*(β_m*Ri - 1))*temp                    # Eq. 19 in Ref. Byun1990
    L_mo = z_b/ζ                                             # LHS of Eq. 3 in Byun1990
    ζ_0 = z_0/L_mo
    ψ_m = ψ_m_stable(ζ, ζ_0, β_m)
    ψ_h = ψ_h_stable(ζ, ζ_0, β_h)
  else
    Q_b = 1/9 * (1 /(γ_m^2) + 3 * γ_h/γ_m * s_b^2)           # Eq. 31 in Ref. Byun1990
    P_b = 1/54 * (-2/(γ_m^3) + 9/γ_m * (-γ_h/γ_m + 3)*s_b^2) # Eq. 32 in Ref. Byun1990
    crit = Q_b^3 - P_b^2
    if crit < 0
      T_b = cbrt(sqrt(-crit) + abs(P_b))                     # Eq. 34 in Ref. Byun1990
      ζ = zfactor * (1/(3*γ_m)-(T_b + Q_b/T_b))              # Eq. 29 in Ref. Byun1990
    else
      θ_b = acos(P_b/sqrt(Q_b^3))                            # Eq. 33 in Ref. Byun1990
      ζ = zfactor * (-2 * sqrt(Q_b) * cos(θ_b/3)+1/(3*γ_m))  # Eq. 28 in Ref. Byun1990
    end
    L_mo = z_b/ζ                                             # LHS of Eq. 3 in Byun1990
    ζ_0 = z_0/L_mo
    ψ_m = ψ_m_unstable(ζ, ζ_0, γ_m)
    ψ_h = ψ_h_unstable(ζ, ζ_0, γ_h)
  end
  cu = k_Karman/(logz-ψ_m)                  # Eq. 10 in Ref. Byun1990, solved for u^*
  cth = k_Karman/(logz-ψ_h)/Pr_0            # Eq. 11 in Ref. Byun1990, solved for h^*
  C_D = cu^2                                                 # Eq. 36 in Byun1990
  C_H = cu*cth                                               # Eq. 37 in Byun1990
  return C_D, C_H, L_mo
end

end # Byun1990 module

module Nishizawa2018
using ...RootSolvers
using ...MoistThermodynamics
using ...PlanetParameters

""" Computes R_z0 expression, defined after Eq. 15 Ref. Nishizawa2018 """
compute_R_z0(z_0, Δz) = 1 - z_0/Δz

""" Computes f_m in Eq. A7 Ref. Nishizawa2018 """
compute_f_m(ζ) = sqrt(sqrt(1-15*ζ))

""" Computes f_h in Eq. A8 Ref. Nishizawa2018 """
compute_f_h(ζ) = sqrt(1-9*ζ)

""" Computes ψ_m in Eq. A3 Ref. Nishizawa2018 """
function compute_ψ_m(ζ, L, a)
  f_m = compute_f_m(ζ)
  return L>=0 ? -a*ζ : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2
end

""" Computes ψ_h in Eq. A4 Ref. Nishizawa2018 """
compute_ψ_h(ζ, L, a, Pr) = L>=0 ? -a*ζ/Pr : 2*log((1+compute_f_h(ζ))/2)

""" Computes Ψ_m in Eq. A5 Ref. Nishizawa2018 """
function compute_Ψ_m(ζ, L, a, tol)
  if ζ < tol
    return ζ>=0 ? -a*ζ/2 : -15*ζ/8 # Computes Ψ_m in Eq. A13 Ref. Nishizawa2018
  else
    f_m = compute_f_m(ζ)
    # Note that "1-f^3" in Ref. Nishizawa2018 is a typo, it is supposed to be "1-f_m^3".
    # This was confirmed by communication with the author.
    return L>=0 ? -a*ζ/2 : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2-1+(1-f_m^3)/(12*ζ)
  end
end

""" Computes Ψ_h in Eq. A6 Ref. Nishizawa2018 """
function compute_Ψ_h(ζ, L, a, Pr, tol)
  if ζ < tol
    return ζ>=0 ? -a*ζ/(2*Pr) : -9*ζ/4 # Computes Ψ_h in Eq. A14 Ref. Nishizawa2018
  else
    f_h = compute_f_h(ζ)
    return L>=0 ? -a*ζ/(2*Pr) : 2*log((1+f_h)/2) + 2*(1-f_h)/(9*ζ)
  end
end

"""
    compute_MO_len(u, θ, flux)

Computes Monin-Obukhov length. Eq. 3 Ref. Nishizawa2018
"""
compute_MO_len(u, θ, flux) = - u^3* θ / (k_Karman * grav * flux)

"""
    compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)

Computes friction velocity, in Eq. 12 in
Ref. Nishizawa2018, by solving the
non-linear equation:

`u_ave = ustar/κ * ( ln(Δz/z_0) - Ψ_m(Δz/L) + z_0/Δz * Ψ_m(z_0/L) + R_z0 [ψ_m(z_0/L) - 1] )`

where `L` is a non-linear function of `ustar` (see `compute_MO_len`).
"""
function compute_friction_velocity(u_ave, θ, flux, Δz, z_0, a, Ψ_m_tol, tol_abs, iter_max)
  ustar_0 = u_ave * k_Karman / log(Δz / z_0)
  ustar = ustar_0
  let u_ave=u_ave, θ=θ, flux=flux, Δz=Δz, z_0=z_0, a=a, Ψ_m_tol=Ψ_m_tol, tol_abs=tol_abs, iter_max=iter_max
    # Note the lowercase psi (ψ) and uppercase psi (Ψ):
    Ψ_m_closure(ζ, L) = compute_Ψ_m(ζ, L, a, Ψ_m_tol)
    ψ_m_closure(ζ, L) = compute_ψ_m(ζ, L, a)
    function compute_u_ave_over_ustar(u)
      L = compute_MO_len(u, θ, flux)
      R_z0 = compute_R_z0(z_0, Δz)
      temp1 = log(Δz/z_0)
      temp2 = - Ψ_m_closure(Δz/L, L)
      temp3 = z_0/Δz * Ψ_m_closure(z_0/L, L)
      temp4 = R_z0 * (ψ_m_closure(z_0/L, L) - 1)
      return (temp1+temp2+temp3+temp4) / k_Karman
    end
    compute_ustar(u) = u_ave/compute_u_ave_over_ustar(u)
    ustar_1 = compute_ustar(ustar_0)
    sol = RootSolvers.find_zero(
      u -> u_ave - u*compute_u_ave_over_ustar(u),
      ustar_0, ustar_1, SecantMethod(), CompactSolution(),
      tol_abs, iter_max)
    ustar = sol.root
  end
  return ustar
end

"""
    compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ, flux, Pr)

Computes exchange transfer coefficients:

  - K_D  momentum exchange coefficient
  - K_H  thermodynamic exchange coefficient
  - L_mo Monin-Obukhov length
"""
function compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ, flux, Pr)

  L_mo = compute_MO_len(u_star, θ, flux)
  ψ_m = compute_ψ_m(z/L_mo, L_mo, a)
  ψ_h = compute_ψ_h(z/L_mo, L_mo, a, Pr)

  K_m = -F_m*k_Karman*z/(u_star * ψ_m) # Eq. 19 in Ref. Nishizawa2018
  K_h = -F_h*k_Karman*z/(Pr * θ * ψ_h) # Eq. 20 in Ref. Nishizawa2018

  return K_m, K_h, L_mo
end

end # Nishizawa2018 module

end # SurfaceFluxes module
