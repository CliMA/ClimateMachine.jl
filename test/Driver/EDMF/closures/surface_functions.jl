
function compute_ustar(
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
    
return ustar
end;


function compute_MO_len(κ::FT, ustar::FT, bflux::FT) where {FT<:Real, PS}
  return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar/bflux/κ
end

function compute_windspeed(u::FT, v::IT, windspeed_min::FT) where {FT}
  return max(hypot(u, v, windspeed_min))
end

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
  windspeed = compute_windspeed(q, k_1, gm, 0.0)^2
  _grav::FT = grav(param_set)

  # test if we need to look at the free convective limit
  z = grid.zc
  h = 0
  Ri_bulk, Ri_bulk_low = 0, 0
  ts = ActiveThermoState(param_set, q, tmp, k_1, gm)
  θ_ρ_b = virtual_pottemp(ts)
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
      Ri_bulk = _grav * (tmp[:θ_ρ, k] - θ_ρ_b) * z[k]/θ_ρ_b / (q[:u, k, gm]^2 + q[:v, k, gm]^2)
      if Ri_bulk > Ri_bulk_crit
        k_star = k
        break
      end
    end
    h = (z[k_star] - z[k_star-1])/(Ri_bulk - Ri_bulk_low) * (Ri_bulk_crit - Ri_bulk_low) + z[k_star-1]
  end
  return h
end
