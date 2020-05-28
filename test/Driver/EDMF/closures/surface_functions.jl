#### Surface model kernels

function compute_inversion_height(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    δ::FT,
    εt::FT,
    ) where {FT, N}

  gm = state
  en = state
  up = state.edmf.updraft
  gm_s = source
  en_s = source
  up_s = source.edmf.updraft
  en_d = state.edmf.environment.diffusive

  # k_1 = first_interior(grid, Zmin()) - how to compute the height of the first grid point
  ts = PhaseEquil(param_set ,gm.e_int, gm.ρ, gm.q_tot)
  windspeed = compute_windspeed(k_1, gm, FT(0.0))^2
  _grav::FT = grav(param_set)
  θ_ρ = 
  θ_ρ_b = virtual_pottemp(ts)
  z = gm_a.z

  # test if we need to look at the free convective limit
  h = 0
  Ri_bulk, Ri_bulk_low = 0, 0
  k_star = k_1
  if windspeed <= SurfaceModel.tke_tol
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
      Ri_bulk = _grav * (θ_ρ - θ_ρ_b) * z[k]/θ_ρ_b / (q[:u, k, gm]^2 + q[:v, k, gm]^2)
      if Ri_bulk > Ri_bulk_crit
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k_star-1]
  end
  return h
end

function compute_blux(m::SurfaceModel, _grav::FT, ϵ_v::FT) # pass in: model and param_set, gm ?
    _T0::FT       = T_0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    ϵ_v::FT       = 1 / molmass_ratio(param_set)
    _grav::FT     = grav(param_set)
    ts = PhaseEquil(param_set ,gm.e_int, gm.ρ, gm.q_tot)
    _cv_m::FT    = cv_m(ts)
  return  _grav*( (m.e_int_surface_flux, -m.q_tot_surface_flux*_e_int_i0 )/(_cv_m*_T0 + gm.e_int - gm.q_tot*_e_int_i0 ) 
                + ( (ϵ_v-1)*m.q_tot_surface_flux)/(1+(ϵ_v-1)*gm.q_tot)) # this equation should verified in the design docs 
end;

function surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, oblength::FT) where FT<:Real
  c_star1 = -flux1/ustar
  c_star2 = -flux2/ustar
  if oblength < 0
    return 4 * c_star1 * c_star2 * (1 - 8.3 * zLL/oblength)^(-2/3)
  else
    return 4 * c_star1 * c_star2
  end
end;
