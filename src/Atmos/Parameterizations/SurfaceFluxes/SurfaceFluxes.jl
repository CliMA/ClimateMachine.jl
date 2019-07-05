"""
    SurfaceFluxes

## Interface
  - [`surface_conditions`](@ref) computes
    - Monin-Obukhov length
    - Potential temperature flux (if not given) using Monin-Obukhov theory
    - transport fluxes using Monin-Obukhov theory
    - friction velocity/temperature scale/tracer scales
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
  L_MO::T
  pottemp_flux_star::T
  flux::Vector{T}
  x_star::Vector{T}
  K_exchange::Vector{T}
end

function Base.show(io::IO, sfc::SurfaceFluxConditions)
  println(io, "----------------------- SurfaceFluxConditions")
  println(io, "L_MO              = ", sfc.L_MO)
  println(io, "pottemp_flux_star = ", sfc.pottemp_flux_star)
  println(io, "flux              = ", sfc.flux)
  println(io, "x_star            = ", sfc.x_star)
  println(io, "K_exchange        = ", sfc.K_exchange)
  println(io, "-----------------------")
end

struct Momentum end
struct Heat end

"""
    surface_conditions(x_initial, x_ave, x_s, z_0, F_exchange, dimensionless_number, θ_bar, Δz, z, a)

Surface conditions given
 - `x_initial` initial guess for solution
      - L_MO
      - u_star
      - θ_star
      - ϕ_star
      - ...
 - `x_ave` volume-averaged value for variable `x`
 - `x_s` surface value for variable `x`
 - `z_0` roughness length for variable `x`
 - `F_exchange` flux at the top for variable `x`
 - `dimensionless_number` dimensionless turbulent transport coefficient:
      - Momentum: 1
      - Heat: Turbulent Prantl number at neutral stratification
      - Mass: Turbulent Schmidt number
      - ...
 - `θ_bar` basic potential temperature
 - `Δz` layer thickness (not spatial discretization)
 - `z` coordinate axis
 - `a` free model parameter with prescribed value of 4.7
 - `pottemp_flux_given` potential temperature flux (optional)

If `pottemp_flux` is not given, then it is computed by iteration
of equations 3, 17, and 18 in Nishizawa2018. Otherwise
"""
function surface_conditions(x_initial::Vector{T},
                            x_ave::Vector{T},
                            x_s::Vector{T},
                            z_0::Vector{T},
                            F_exchange::Vector{T},
                            dimensionless_number::Vector{T},
                            θ_bar::T,
                            Δz::T,
                            z::T,
                            a::T,
                            pottemp_flux_given::Union{Nothing,T}=nothing
                            ) where T<:AbstractFloat

  n_vars = length(x_initial)-1
  @assert length(x_initial)==n_vars+1
  @assert length(x_ave)==n_vars
  @assert length(x_s)==n_vars
  @assert length(z_0)==n_vars
  @assert length(F_exchange)==n_vars
  @assert length(dimensionless_number)==n_vars
  function f!(F, x_all)
      L_MO, x_vec = x_all[1], x_all[2:end]
      u, θ = x_vec[1], x_vec[2]
      pottemp_flux = pottemp_flux_given==nothing ? - u * θ : pottemp_flux_given
      F[1] = L_MO - compute_MO_len(u, θ_bar, pottemp_flux)
      for i in 1:n_vars
        ϕ = x_vec[i]
        transport = i==1 ? Momentum() : Heat()
        F[i+1] = ϕ - compute_physical_scale(ϕ, u, Δz, a, θ_bar, pottemp_flux, z_0[i], x_ave[i], x_s[i], dimensionless_number[i], transport)
      end
  end
  sol = nlsolve(f!, x_initial, autodiff = :forward)
  if converged(sol)
    L_MO, x_star = sol.zero[1], sol.zero[2:end]
    u_star, θ_star = x_star[1], x_star[2]
  else
    @show sol
    error("Unconverged surface fluxes")
  end

  pottemp_flux_star = -u_star^3*θ_bar/(k_Karman*grav*L_MO)
  flux = -u_star*x_star
  K_exchange = compute_exchange_coefficients(z, F_exchange, a, x_star, θ_bar, dimensionless_number, L_MO)

  return SurfaceFluxConditions(L_MO,
                               pottemp_flux_star,
                               flux,
                               x_star,
                               K_exchange)
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
    compute_exchange_coefficients(z, F_exchange, a, x_star, θ_bar, dimensionless_number, L_MO)

Computes exchange transfer coefficients

  - `K_exchange` exchange coefficients
"""
function compute_exchange_coefficients(z, F_exchange, a, x_star, θ_bar, dimensionless_number, L_MO)

  N = length(F_exchange)
  K_exchange = Vector{typeof(z)}(undef, N)
  for i in 1:N
    transport = i==1 ? Momentum() : Heat()
    psi = compute_psi(z/L_MO, L_MO, a, dimensionless_number[i], transport)
    K_exchange[i] = -F_exchange[i]*k_Karman*z/(x_star[i] * psi) # Eq. 19 in Ref. Nishizawa2018
  end

  return K_exchange
end

end # SurfaceFluxes module
