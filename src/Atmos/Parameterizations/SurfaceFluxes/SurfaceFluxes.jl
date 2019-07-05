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

  - Ref. Nishizawa2018:
    - Nishizawa, S., and Y. Kitamura. "A Surface Flux Scheme Based on the
      Monin-Obukhov Similarity for Finite Volume Models." Journal of
      Advances in Modeling Earth Systems 10.12 (2018): 3159-3175.
      https://doi.org/10.1029/2018MS001534

  - Ref. Wyngaard1975:
    - Wyngaard, John C. "Modeling the planetary boundary layer-Extension
      to the stable case." Boundary-Layer Meteorology 9.4 (1975): 441-460.
      https://link.springer.com/article/10.1007/BF00223393

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
using NLsolve

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
  pottemp_flux::T
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
  println(io, "pottemp_flux            = ", sfc.pottemp_flux)
  println(io, "exchange_coeff_momentum = ", sfc.exchange_coeff_momentum)
  println(io, "exchange_coeff_heat     = ", sfc.exchange_coeff_heat)
  println(io, "-----------------------")
end

struct Momentum end
struct Heat end

"""
    surface_conditions(u_ave::DT,
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
                       dimensionless_number::DT,
                       pottemp_flux::DT
                       ) where DT<:AbstractFloat

Surface conditions given
 - `u_ave` volume-averaged horizontal wind speed
 - `θ_ave` volume-averaged potential temperature
 - `θ_bar` basic potential temperature
 - `θ_s` surface potential temperature
 - `Δz` layer thickness (not spatial discretization)
 - `z_0_m` roughness length for momentum
 - `z_0_h` roughness length for heat
 - `z` coordinate axis
 - `F_m` momentum flux at the top
 - `F_h` heat flux at the top
 - `a` free model parameter with prescribed value of 4.7
 - `dimensionless_number` Prantl number at neutral stratification
 - `pottemp_flux` potential temperature flux (optional)

If `pottemp_flux` is not given, then it is computed by iteration
of equations 3, 17, and 18 in Nishizawa2018.
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
                            dimensionless_number::DT) where DT<:AbstractFloat

  initial_x = [100, 15.0, 350.0] # TODO: improve initial guess
  function f!(F, x)
      L_MO, u, θ = x[1], x[2], x[3]
      pottemp_flux = - u * θ
      F[1] = L_MO - compute_MO_len(u, θ_bar, pottemp_flux)
      F[2] = u - compute_physical_scale(u, u, Δz, a, θ_bar, pottemp_flux, z_0_m, u_ave, 0, 1, Momentum())
      F[3] = θ - compute_physical_scale(θ, u, Δz, a, θ_bar, pottemp_flux, z_0_h, θ_ave, θ_s, dimensionless_number, Heat())
  end
  sol = nlsolve(f!, initial_x, autodiff = :forward)
  if converged(sol)
    L_MO, u_star, θ_star = sol.zero
  else
    error("Unconverged surface fluxes")
  end
  pottemp_flux_star = -u_star^3*θ_bar/(k_Karman*grav*L_MO)

  momentum_flux = -u_star^2
  buoyancy_flux = -u_star*θ_star
  K_m, K_h, L_MO = compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, dimensionless_number, pottemp_flux_star)

  return SurfaceFluxConditions(L_MO,
                               momentum_flux,
                               buoyancy_flux,
                               u_star,
                               θ_star,
                               pottemp_flux_star,
                               K_m,
                               K_h)
end


function compute_physical_scale(x, u, Δz, a, θ_bar, pottemp_flux, z_0, x_ave, x_s, dimensionless_number, transport)
  L = compute_MO_len(u, θ_bar, pottemp_flux)
  R_z0 = compute_R_z0(z_0, Δz)
  temp1 = log(Δz/z_0)
  temp2 = - compute_Psi(Δz/L, L, a, dimensionless_number, transport)
  temp3 = z_0/Δz * compute_Psi(z_0/L, L, a, dimensionless_number, transport)
  temp4 = R_z0 * (compute_psi(z_0/L, L, a, dimensionless_number, transport) - 1)
  return (1/dimensionless_number)*k_Karman/(temp1+temp2+temp3+temp4)*(x_ave-x_s)
end

""" Computes R_z0 expression, defined after Eq. 15 Ref. Nishizawa2018 """
compute_R_z0(z_0, Δz) = 1 - z_0/Δz

""" Computes f_m in Eq. A7 Ref. Nishizawa2018 """
compute_f_m(ζ) = sqrt(sqrt(1-15*ζ))

""" Computes f_h in Eq. A8 Ref. Nishizawa2018 """
compute_f_h(ζ) = sqrt(1-9*ζ)

""" Computes psi_m in Eq. A3 Ref. Nishizawa2018 """
function compute_psi(ζ, L, a, dimensionless_number, ::Momentum)
  f_m = compute_f_m(ζ)
  return L>=0 ? -a*ζ : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2
end

""" Computes psi_h in Eq. A4 Ref. Nishizawa2018 """
compute_psi(ζ, L, a, dimensionless_number, ::Heat) = L>=0 ? -a*ζ/dimensionless_number : 2*log((1+compute_f_h(ζ))/2)

""" Computes Psi_m in Eq. A5 Ref. Nishizawa2018 """
function compute_Psi(ζ, L, a, dimensionless_number, ::Momentum)
  if abs(ζ) < eps(typeof(L))
    return ζ>=0 ? -a*ζ/2 : -15*ζ/8 # Computes Psi_m in Eq. A13 Ref. Nishizawa2018
  else
    f_m = compute_f_m(ζ)
    # Note that "1-f^3" in Ref. Nishizawa2018 is a typo, it is supposed to be "1-f_m^3".
    # This was confirmed by communication with the author.
    return L>=0 ? -a*ζ/2 : log((1+f_m)^2*(1+f_m^2)/8) - 2*atan(f_m)+π/2-1+(1-f_m^3)/(12*ζ)
  end
end

""" Computes Psi_h in Eq. A6 Ref. Nishizawa2018 """
function compute_Psi(ζ, L, a, dimensionless_number, ::Heat)
  if abs(ζ) < eps(typeof(L))
    return ζ>=0 ? -a*ζ/(2*dimensionless_number) : -9*ζ/4 # Computes Psi_h in Eq. A14 Ref. Nishizawa2018
  else
    f_h = compute_f_h(ζ)
    return L>=0 ? -a*ζ/(2*dimensionless_number) : 2*log((1+f_h)/2) + 2*(1-f_h)/(9*ζ)
  end
end

"""
    compute_MO_len(u, θ, flux)

Computes Monin-Obukhov length. Eq. 3 Ref. Nishizawa2018
"""
compute_MO_len(u, θ_bar, flux) = - u^3* θ_bar / (k_Karman * grav * flux)

"""
    compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, dimensionless_number, pottemp_flux)

Computes exchange transfer coefficients

  - `K_D`  momentum exchange coefficient
  - `K_H`  thermodynamic exchange coefficient
  - `L_MO` Monin-Obukhov length
"""
function compute_exchange_coefficients(z, F_m, F_h, a, u_star, θ_star, θ_bar, dimensionless_number, pottemp_flux)

  L_MO = compute_MO_len(u_star, θ_bar, pottemp_flux)
  psi_m = compute_psi(z/L_MO, L_MO, a, 1, Momentum())
  psi_h = compute_psi(z/L_MO, L_MO, a, dimensionless_number, Heat())

  K_m = -F_m*k_Karman*z/(u_star * psi_m) # Eq. 19 in Ref. Nishizawa2018
  K_h = -F_h*k_Karman*z/(dimensionless_number * θ_star * psi_h) # Eq. 20 in Ref. Nishizawa2018

  return K_m, K_h, L_MO
end

end # SurfaceFluxes module
