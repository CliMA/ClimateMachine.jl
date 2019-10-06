#### Surface functions

using CLIMA.SurfaceFluxes.Nishizawa2018
using Distributions
using Statistics

export update_surface!

"""
    update_surface!

Update surface conditions including
 - `windspeed`
 - `ρq_tot_flux`
 - `ρθ_liq_flux`
 - `bflux`
 - `obukhov_length`
 - `rho_uflux`
 - `rho_vflux`
"""
function update_surface! end

function update_surface!(tmp::StateVec, q::StateVec, grid::Grid{FT}, params, case::BOMEX) where FT
  gm, en, ud, sd, al = allcombinations(tmp)
  k_1 = first_interior(grid, Zmin())
  z_1 = grid.zc[k_1]
  ρ_0_surf = air_density(params[:Tg], params[:Pg], PhasePartition(params[:qtg]))
  α_0_surf = 1/ρ_0_surf
  T_1 = tmp[:T, k_1, gm]
  θ_liq_1 = q[:θ_liq, k_1, gm]
  q_tot_1 = q[:q_tot, k_1, gm]
  V_1 = q[:v, k_1, gm]
  U_1 = q[:u, k_1, gm]

  rho_tflux =  params[:shf] /(cp_m(PhasePartition(params[:qsurface])))

  params[:windspeed] = compute_windspeed(q, k_1, FT(0.0))
  params[:ρq_tot_flux] = params[:lhf]/(latent_heat_vapor(params[:Tsurface]))
  params[:ρθ_liq_flux] = rho_tflux / exner(params[:Pg])
  params[:bflux] = buoyancy_flux(params[:shf], params[:lhf], T_1, q_tot_1, α_0_surf)

  params[:obukhov_length] = compute_MO_len(params[:ustar], params[:bflux])
  params[:rho_uflux] = - ρ_0_surf *  params[:ustar] * params[:ustar] / params[:windspeed] * U_1
  params[:rho_vflux] = - ρ_0_surf *  params[:ustar] * params[:ustar] / params[:windspeed] * V_1
end


function percentile_bounds_mean_norm(low_percentile::FT, high_percentile::FT, n_samples::I) where {FT<:Real, I}
    x = rand(Normal(), n_samples)
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low<y<xp_high, x)
    return Statistics.mean(x)
end

"""
    surface_tke(ustar::FT, wstar::FT, zLL::FT, obukhov_length::FT) where FT<:Real

computes the surface tke

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `zLL` elevation at the first grid level
 - `obukhov_length` Monin-Obhukov length
"""
function surface_tke(ustar::FT, wstar::FT, zLL::FT, obukhov_length::FT) where FT<:Real
  if obukhov_length < 0
    return ((3.75 + cbrt(zLL/obukhov_length * zLL/obukhov_length)) * ustar * ustar + 0.2 * wstar * wstar)
  else
    return (3.75 * ustar * ustar)
  end
end

"""
    surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, oblength::FT) where FT<:Real

computes the surface variance given

 - `ustar` friction velocity
 - `wstar` convective velocity
 - `zLL` elevation at the first grid level
 - `oblength` Monin-Obhukov length
"""
function surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, oblength::FT) where FT<:Real
  c_star1 = -flux1/ustar
  c_star2 = -flux2/ustar
  if oblength < 0
    return 4 * c_star1 * c_star2 * (1 - 8.3 * zLL/oblength)^(-2/3)
  else
    return 4 * c_star1 * c_star2
  end
end

"""
    compute_convective_velocity(bflux, inversion_height)

Computes the convective velocity scale, given the buoyancy flux
`bflux`, and inversion height `inversion_height`.
FIXME: add reference
"""
compute_convective_velocity(bflux::FT, inversion_height::FT) where FT = cbrt(max(bflux * inversion_height, FT(0)))

"""
    compute_windspeed(q::StateVec, k::I, windspeed_min::FT)

Computes the windspeed
"""
function compute_windspeed(q::StateVec, k::I, windspeed_min::FT) where {FT, I}
  gm, en, ud, sd, al = allcombinations(q)
  return max(sqrt(q[:u, k, gm]^2 + q[:v, k, gm]^2), windspeed_min)
end

"""
    compute_inversion_height(tmp::StateVec, q::StateVec, grid::Grid, params)

Computes the inversion height (a non-local variable)
FIXME: add reference
"""
function compute_inversion_height(tmp::StateVec, q::StateVec, grid::Grid, params)
  @unpack params Ri_bulk_crit tke_surface_tol
  gm, en, ud, sd, al = allcombinations(q)
  k_1 = first_interior(grid, Zmin())
  windspeed = compute_windspeed(q, k_1, 0.0)^2

  # test if we need to look at the free convective limit
  z = grid.zc
  h = 0
  Ri_bulk, Ri_bulk_low = 0, 0
  ts = ActiveThermoState(q, tmp, k_1, gm)
  θ_ρ_b = virtual_pottemp(ts)
  k_star = k_1
  if windspeed <= tke_surface_tol
    for k in over_elems_real(grid)
      if tmp[:θ_ρ, k] > θ_ρ_b
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(tmp[:θ_ρ, k_star] - tmp[:θ_ρ, k_star-1]) * (θ_ρ_b - tmp[:θ_ρ, k_star-1]) + z[k_star-1]
  else
    for k in over_elems_real(grid)
      Ri_bulk_low = Ri_bulk
      Ri_bulk = grav * (tmp[:θ_ρ, k] - θ_ρ_b) * z[k]/θ_ρ_b / (q[:u, k, gm]^2 + q[:v, k, gm]^2)
      if Ri_bulk > Ri_bulk_crit
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k_star-1]
  end
  return h
end

"""
    compute_MO_len(ustar::FT, bflux::FT) where {FT<:Real}

Compute Monin-Obhukov length given
 - `ustar` friction velocity
 - `bflux` buoyancy flux
"""
function compute_MO_len(ustar::FT, bflux::FT) where {FT<:Real}
  return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / k_Karman
end
