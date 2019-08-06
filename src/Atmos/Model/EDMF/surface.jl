
#### Surface model
abstract type AbstractSurfaceModel{T} end
export SurfaceModel

using ..SurfaceFluxes

function update_aux!(edmf::EDMF{N}, ::AbstractSurfaceModel, state::Vars, diffusive::Vars, aux::Vars, t::Real) where N; end

vars_state(    ::AbstractSurfaceModel, T) = @vars()
vars_gradient( ::AbstractSurfaceModel, T) = @vars()
vars_diffusive(::AbstractSurfaceModel, T) = @vars()
vars_aux(      ::AbstractSurfaceModel, T) = @vars()
vars_inputs(   ::AbstractSurfaceModel, T) = @vars()

struct SurfaceModel{T} <: AbstractSurfaceModel{T}
  TCV_w_e_int::T
  TCV_w_q_tot::T
  exch_coeff_e_int::T
  exch_coeff_q_tot::T
  surface_scalar_coeff::T
  x_initial::Vector{T}
  x_ave::Vector{T}
  x_s::Vector{T}
  z_0::Vector{T}
  F_exchange::Vector{T}
  dimensionless_number::Vector{T}
  θ_bar::T
  Δz::T
  z::T
  a::T
end
function SurfaceModel(::Type{DT}) where DT
  TCV_w_e_int = DT(1.0)
  TCV_w_q_tot = DT(1.0)
  exch_coeff_e_int = DT(1.0)
  exch_coeff_q_tot = DT(1.0)
  surface_scalar_coeff = DT(1.0)
  θ_bar = DT(1.0)
  Δz = DT(1.0)
  z = DT(1.0)
  a = DT(1.0)
  x_initial = DT[1.0, 1.0]
  x_ave = DT[1.0, 1.0]
  x_s = DT[1.0, 1.0]
  z_0 = DT[1.0, 1.0]
  F_exchange = DT[1.0, 1.0]
  dimensionless_number = DT[1.0, 1.0]
  return SurfaceModel{DT}(TCV_w_e_int, TCV_w_q_tot, exch_coeff_e_int,
                          exch_coeff_q_tot, surface_scalar_coeff,
                          x_initial, x_ave, x_s, z_0, F_exchange,
                          dimensionless_number, θ_bar, Δz, z, a)
end


function surface_fluxes(edmf::EDMF{N}, m::SurfaceModel, state::Vars, aux::Vars) where N
  id = idomains(N)
  windspeed = horizontal_windspeed(m)
  ts = thermo_state(edmf, state, aux)
  args = m.x_initial, m.x_ave, m.x_s, m.z_0, m.F_exchange, m.dimensionless_number, m.θ_bar, m.Δz, m.z, m.a
  sfc = surface_conditions(args...)

  # z_L1 = z_first_interior(aux) # need functionality
  inversion_height = aux.inversion_height # computing this requires argmax_z(BulkRichardsonNumber(z)), and is a single value per column
  wstar = convective_velocity_scale(sfc.bflux, inversion_height)

  tke_surf        = surface_tke(ustar, z_L1, sfc.L_MO, wstar)
  var_q_tot       = surface_variance(  ustar, z_L1, sfc.L_MO, m.TCV_w_q_tot, m.TCV_w_q_tot)
  var_e_int       = surface_variance(  ustar, z_L1, sfc.L_MO, m.TCV_w_e_int, m.TCV_w_e_int)
  var_e_int_q_tot = surface_variance(  ustar, z_L1, sfc.L_MO, m.TCV_w_e_int, m.TCV_w_q_tot)

  e_int_flux = m.TCV_w_e_int .+ m.surface_scalar_coeff*m.exch_coeff_e_int*windspeed*sqrt(var_e_int)
  q_tot_flux = m.TCV_w_q_tot .+ m.surface_scalar_coeff*m.exch_coeff_q_tot*windspeed*sqrt(var_q_tot)

  return e_int_flux, q_tot_flux, var_q_tot, var_e_int, var_e_int_q_tot, tke_surf, windspeed
end


#### pure functions

using Statistics
function percentile_bounds_mean_norm(low_percentile::R, high_percentile::R, n_samples::R) where R
    x = rand(Normal(), n_samples)
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low<y<xp_high, x)
    return Statistics.mean(x)
end

"""
    surface_tke(ustar, wstar, z_L1, MoninObhukovLen)

The surface turbulent kinetic energy

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `z_L1` elevation at the first grid level
 - `MoninObhukovLen` Monin-Obhukov length
"""
function surface_tke(ustar::R, z_L1::R, MoninObhukovLen::R, wstar::R) where R
  if MoninObhukovLen < 0
    return 3.75 * ustar^2 + 0.2 * wstar^2 + ustar^2*cbrt((z_L1/MoninObhukovLen)^2)
  else
    return 3.75 * ustar^2
  end
end

"""
    surface_variance(ustar, z_L1, MoninObhukovLen, flux1, flux2)

The surface variance given

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `z_L1` elevation at the first grid level
 - `MoninObhukovLen` Monin-Obhukov length
"""
function surface_variance(ustar::R, z_L1::R, MoninObhukovLen::R, flux1::R, flux2::R) where R
  if MoninObhukovLen < 0
    return 4 * flux1*flux2/(ustar^2) * (1 - 8.3 * z_L1/MoninObhukovLen)^(-2/3)
  else
    return 4 * flux1*flux2/(ustar^2)
  end
end

"""
    convective_velocity_scale(bflux::R, inversion_height::R) where R

The convective velocity scale, given:
 `bflux` buoyancy flux
 `inversion_height` inversion height
FIXME: add reference
"""
convective_velocity_scale(bflux::R, inversion_height::R) where R = cbrt(max(bflux * inversion_height, R(0)))
