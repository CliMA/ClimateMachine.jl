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

function compute_blux(
    ss::SingleStack{FT, N},
    m::SurfaceModel,
    source::Vars,
    state::Vars,
    ) where {FT, N}

    ts = PhaseEquil(param_set ,state.e_int, state.ρ, state.q_tot)
    ϵ_v::FT       = 1 / molmass_ratio(param_set)
    _T0::FT       = T_0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    _grav::FT     = grav(param_set)
    _cv_m::FT    = cv_m(ts)
  return  _grav*( (m.e_int_surface_flux, -m.q_tot_surface_flux*_e_int_i0 )/(_cv_m*_T0 + state.e_int - state.q_tot*_e_int_i0 ) 
                + ( (ϵ_v-1)*m.q_tot_surface_flux)/(1+(ϵ_v-1)*state.q_tot)) # this equation should verified in the design docs 
end;

function compute_MO_len(κ::FT, ustar::FT, bflux::FT) where {FT<:Real, PS}
  return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / κ
end;

# function env_surface_covariances(e_int_surface_flux::FT, q_tot_surface_flux::FT, ustar::FT, zLL::FT, oblength::FT) where FT<:Real
function env_surface_covariances(ss::SingleStack{FT, N},
    m::SurfaceModel,
    edmf::EDMF{FT,N},
    source::Vars,
    state::Vars,
    ) where {FT, N}
  # yair - I would like to call the surface functions from src/Atmos/Model/SurfaceFluxes.jl
  bflux = Nishizawa2018.compute_buoyancy_flux(param_set, m.shf, m.lhf, T_b, q, α_0)
  oblength = Nishizawa2018.monin_obukhov_len(param_set, ustar, bflux)
  zLL = FT(20) # how to get the z fir st interior ?
  if oblength < 0
    e_int_var       = FT(4) * (edmf.e_int_surface_flux*edmf.e_int_surface_flux)/(ustar*ustar) * (FT(1) - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    q_tot_var       = FT(4) * (edmf.q_tot_surface_flux*edmf.q_tot_surface_flux)/(ustar*ustar) * (FT(1) - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    e_int_q_tot_cov = FT(4) * (edmf.e_int_surface_flux*edmf.q_tot_surface_flux)/(ustar*ustar) * (FT(1) - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    tke             = ((FT(3.75) + cbrt(zLL/obukhov_length * zLL/obukhov_length)) * ustar * ustar)
    return e_int_var, q_tot_var, e_int_q_tot_cov, tke
  else
    e_int_var       = FT(4) * (edmf.e_int_surface_flux * edmf.e_int_surface_flux)/(ustar*ustar)
    q_tot_var       = FT(4) * (edmf.q_tot_surface_flux * edmf.q_tot_surface_flux)/(ustar*ustar)
    e_int_q_tot_cov = FT(4) * (edmf.e_int_surface_flux * edmf.q_tot_surface_flux)/(ustar*ustar)
    tke             = (FT(3.75) * ustar * ustar)
    return e_int_var, q_tot_var, e_int_q_tot_cov, tke
  end
end;