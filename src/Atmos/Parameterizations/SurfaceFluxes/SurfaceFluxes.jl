"""
    SurfaceFluxes

  Surface flux functions, e.g., for buoyancy flux,
  friction velocity, and exchange coefficients.

## Interface
  - [`surface_conditions`](@ref) computes
    - buoyancy flux
    - Monin-Obukhov length
    - friction velocity
    - temperature scale
    - exchange coefficients

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

  - Ref. Businger1971:
    - Businger, Joost A., et al. "Flux-profile relationships in the
      atmospheric surface layer." Journal of the atmospheric Sciences
      28.2 (1971): 181-189.

"""
module SurfaceFluxes

using ..RootSolvers
using ..MoistThermodynamics
using ..PlanetParameters
using DocStringExtensions

export surface_conditions

"""
    SurfaceFluxConditions{T}

Surface flux conditions, returned from `surface_conditions`.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SurfaceFluxConditions{T}
  momentum_flux::T
  buoyancy_flux::T
  Monin_Obukhov_length::T
  friction_velocity::T
  temperature_scale::T
  exchange_coeff_momentum::T
  exchange_coeff_heat::T
end

function Base.show(io::IO, sfc::SurfaceFluxConditions)
  println(io, "----------------------- SurfaceFluxConditions")
  println(io, "momentum_flux           = ", sfc.momentum_flux)
  println(io, "buoyancy_flux           = ", sfc.buoyancy_flux)
  println(io, "Monin_Obukhov_length    = ", sfc.Monin_Obukhov_length)
  println(io, "friction_velocity       = ", sfc.friction_velocity)
  println(io, "temperature_scale       = ", sfc.temperature_scale)
  println(io, "exchange_coeff_momentum = ", sfc.exchange_coeff_momentum)
  println(io, "exchange_coeff_heat     = ", sfc.exchange_coeff_heat)
  println(io, "-----------------------")
end

"""
    surface_conditions(u_ave::DT, θ_bar::DT, Δz::DT, z_0_m::DT, z_0_h::DT,
        z::DT, F_m::DT, F_h::DT, a::DT, θ_s::DT, Pr::DT) where DT<:AbstractFloat

Surface conditions given
 - `u_ave` volume-averaged horizontal wind speed
 - `θ_ave` volume-averaged potential temperature
 - `θ_bar` basic potential temperature
 - `θ_s` surface potential temperature
 - `Δz` layer thickness
 - `z_0_m` roughness length for momentum
 - `z_0_h` roughness length for heat
 - `z` coordinate axis
 - `F_m` momentum flux at the top
 - `F_h` heat flux at the top
 - `a` free model parameter with prescribed value of 4.7
 - `Pr` Prantl number at neutral stratification
 - `pottemp_flux` potential temperature flux
"""
function surface_conditions(u_ave::DT,
                            θ_bar::DT,
                            θ_ave::DT,
                            θ_s::DT,
                            Δz::DT,
                            z_0_m::DT,
                            z_0_h::DT,
                            z::DT,
                            F_m::DT,
                            F_h::DT,
                            a::DT,
                            Pr::DT,
                            pottemp_flux::DT
                            ) where DT<:AbstractFloat

  @assert 0 <= z <= z_0_m
  @assert 0 <= z <= z_0_h

  tol_abs, iter_max = DT(1e-3), 100
  u_star = compute_physical_scale(Δz, a, tol_abs, iter_max, θ_bar, pottemp_flux, z_0_m, u_ave, 0, 1)
  θ_star = compute_physical_scale(Δz, a, tol_abs, iter_max, θ_bar, pottemp_flux, z_0_h, θ_ave, θ_s, 1/Pr)
  momentum_flux = -u_star^2
  buoyancy_flux = -u_star*θ_star
  K_m, K_h, L_MO = compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, Pr, pottemp_flux)

  return SurfaceFluxConditions(L_MO,
                               momentum_flux,
                               buoyancy_flux,
                               u_star,
                               θ_star,
                               K_m,
                               K_h)
end

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
function compute_Ψ_m(ζ, L, a)
  if abs(ζ) < eps(typeof(L))
    return ζ>=0 ? -a*ζ/2 : -15*ζ/8 # Computes Ψ_m in Eq. A13 Ref. Nishizawa2018
  else
    f_m = compute_f_m(ζ)
    # Note that "1-f^3" in Ref. Nishizawa2018 is a typo, it is supposed to be "1-f_m^3".
    # This was confirmed by communication with the author.
    return L>=0 ? -a*ζ/2 : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2-1+(1-f_m^3)/(12*ζ)
  end
end

""" Computes Ψ_h in Eq. A6 Ref. Nishizawa2018 """
function compute_Ψ_h(ζ, L, a, Pr)
  if abs(ζ) < eps(typeof(L))
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
compute_MO_len(u, θ_bar, flux) = - u^3* θ_bar / (k_Karman * grav * flux)


struct Heat end
struct Momentum end


"""
    transport_flux_m(u)

Momentum transport flux
"""
transport_flux(u, ::Momentum) = -u^2

"""
    transport_flux_h(u)

Thermal heat transport flux
"""
transport_flux_h(u, θ) = -u*θ

"""
    compute_physical_scale(Δz, a, tol_abs, iter_max, θ_bar, pottemp_flux, z_0, x_ave, x_s, coeff)

`x_star = κ/denom*(x_ave-x_s)`

where

`denom = ln(Δz/z_0) - Ψ_m(Δz/L) + z_0/Δz * Ψ_m(z_0/L) + R_z0 (ψ_m(z_0/L) - 1)`
"""
function compute_physical_scale(Δz, a, tol_abs, iter_max, θ_bar, pottemp_flux, z_0, x_ave, x_s, coeff)
  x_star_0 = x_ave * k_Karman / log(Δz / z_0)
  x_star = x_star_0
  let x_ave=x_ave, θ_bar=θ_bar, Δz=Δz, z_0=z_0, a=a, tol_abs=tol_abs, iter_max=iter_max
    # Note the lowercase psi (ψ) and uppercase psi (Ψ):
    Ψ_m_closure(ζ, L) = compute_Ψ_m(ζ, L, a)
    ψ_m_closure(ζ, L) = compute_ψ_m(ζ, L, a)
    function compute_x_star(x)
      L = compute_MO_len(x, θ_bar, pottemp_flux)
      R_z0 = compute_R_z0(z_0, Δz)
      temp1 = log(Δz/z_0)
      temp2 = - Ψ_m_closure(Δz/L, L)
      temp3 = z_0/Δz * Ψ_m_closure(z_0/L, L)
      temp4 = R_z0 * (ψ_m_closure(z_0/L, L) - 1)
      return coeff*k_Karman/(temp1+temp2+temp3+temp4)*(x_ave-x_s)
    end
    compute_roots(x) = x - compute_x_star(x)
    x_star_1 = compute_x_star(x_star_0)
    x_star, converged = RootSolvers.find_zero(x -> compute_roots(x), x_star_0, x_star_1, SecantMethod();
                                              xatol=tol_abs, maxiters=iter_max)
  end
  return x_star
end


"""
    compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, Pr)

Computes exchange transfer coefficients:

  - K_D  momentum exchange coefficient
  - K_H  thermodynamic exchange coefficient
  - L_MO Monin-Obukhov length
"""
function compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, Pr, pottemp_flux)

  L_MO = compute_MO_len(u_star, θ_bar, pottemp_flux)
  ψ_m = compute_ψ_m(z/L_MO, L_MO, a)
  ψ_h = compute_ψ_h(z/L_MO, L_MO, a, Pr)

  K_m = -F_m*k_Karman*z/(u_star * ψ_m) # Eq. 19 in Ref. Nishizawa2018
  K_h = -F_h*k_Karman*z/(Pr * θ_star * ψ_h) # Eq. 20 in Ref. Nishizawa2018

  return K_m, K_h, L_MO
end

end # SurfaceFluxes module
