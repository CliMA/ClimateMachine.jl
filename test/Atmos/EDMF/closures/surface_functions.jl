#### Surface model kernels

## --- revert to use compute_buoyancy_flux in SurfaceFluxes.jl ---|

# function compute_blux(
#     ss::SingleStack{FT, N},
#     m::SurfaceModel,
#     source::Vars,
#     state::Vars,
#     ) where {FT, N}

#     ts = PhaseEquil(param_set ,state.e_int, state.ρ, state.q_tot)
#     ϵ_v::FT       = 1 / molmass_ratio(param_set)
#     _T0::FT       = T_0(param_set)
#     _e_int_i0::FT = e_int_i0(param_set)
#     _grav::FT     = grav(param_set)
#     _cv_m::FT    = cv_m(ts)
#   return  _grav*( (m.e_int_surface_flux, -m.q_tot_surface_flux*_e_int_i0 )/(_cv_m*_T0 + state.e_int - state.q_tot*_e_int_i0 )
#                 + ( (ϵ_v-1)*m.q_tot_surface_flux)/(1+(ϵ_v-1)*state.q_tot)) # this equation should verified in the design docs
# end;
# function compute_MO_len(κ::FT, ustar::FT, bflux::FT) where {FT<:Real, PS}
#   return abs(bflux) < FT(1e-10) ? FT(0) : -ustar * ustar * ustar / bflux / κ
# end;

## --- revert to use compute_buoyancy_flux in SurfaceFluxes.jl ---|

function env_surface_covariances(
    ss::SingleStack{FT, N},
    m::SurfaceModel,
    edmf::EDMF{FT,N},
    source::Vars,
    state::Vars,
    ) where {FT, N}
  # yair - I would like to call the surface functions from src/Atmos/Model/SurfaceFluxes.jl
  # bflux = Nishizawa2018.compute_buoyancy_flux(ss.param_set, m.shf, m.lhf, T_b, q, α_0) # missing def of m.shf, m.lhf, T_b, q, α_0
  # oblength = Nishizawa2018.monin_obukhov_len(ss.param_set, u, θ, bflux) # missing def of u, θ,
  # Use fixed values for now
  zLL = FT(20) # how to get the z first interior ?
  if oblength < 0
    e_int_var       = 4 * (edmf.e_int_surface_flux*edmf.e_int_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    q_tot_var       = 4 * (edmf.q_tot_surface_flux*edmf.q_tot_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    e_int_q_tot_cov = 4 * (edmf.e_int_surface_flux*edmf.q_tot_surface_flux)/(ustar*ustar) * (1 - FT(8.3) * zLL/oblength)^(-FT(2)/FT(3))
    tke             = ustar * ustar * (FT(3.75) + cbrt(zLL/obukhov_length * zLL/obukhov_length))
    return e_int_var, q_tot_var, e_int_q_tot_cov, tke
  else
    e_int_var       = 4 * (edmf.e_int_surface_flux * edmf.e_int_surface_flux)/(ustar*ustar)
    q_tot_var       = 4 * (edmf.q_tot_surface_flux * edmf.q_tot_surface_flux)/(ustar*ustar)
    e_int_q_tot_cov = 4 * (edmf.e_int_surface_flux * edmf.q_tot_surface_flux)/(ustar*ustar)
    tke             = ustar * ustar * FT(3.75)
    return e_int_var, q_tot_var, e_int_q_tot_cov, tke
  end
end;

function compute_updraft_surface_BC(
    ss::SingleStack{FT, N},
    m::SurfaceModel,
    edmf::EDMF{FT,N},
    state::Vars,
    ) where {FT, N}
  
  gm = state
  en = state
  up = state.edmf.updraft
  ρinv = 1/gm.ρ
  
  tke, e_int_cv ,q_tot_cv ,e_int_q_tot_cv = env_surface_covariances(ss, m, edmf, state)
  upd_a_surf::SVector{N, FT}
  upd_e_int_surf::SVector{N, FT}
  upd_q_tot_surf::SVector{N, FT}
  for i in 1:N
    surface_scalar_coeff = percentile_bounds_mean_norm(1 - m.surface_area+ i * FT(m.surface_area/N),
                                                        1 - m.surface_area + (i+1)*FT(m.surface_area/N), 1000)
    upd_a_surf[i]     = gm.ρ * FT(m.surface_area/N)
    upd_e_int_surf[i] = gm.ρe_int + surface_scalar_coeff*sqrt(e_int_cv)*gm.ρ
    upd_q_tot_surf[i] = gm.ρq_tot + surface_scalar_coeff*sqrt(q_tot_cv)*gm.ρ
  end

  return upd_a_surf ,upd_e_int_surf ,upd_q_tot_surf
end;

function percentile_bounds_mean_norm(low_percentile::FT, high_percentile::FT, n_samples::Int) where {FT<:Real, I}
    xp_low = quantile(Normal(), low_percentile)
    xp_high = quantile(Normal(), high_percentile)
    filter!(y -> xp_low<y<xp_high, x)
    return Statistics.mean(x)
end
