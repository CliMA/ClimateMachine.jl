#### InitParams

export Params

"""
    Params(::BOMEX)

Initialize stand-alone input parameters to
solve the EDMF equations for the BOMEX case.
"""
function Params(::BOMEX)
  params = Dict()
  params[:N_subdomains] = 3                                           # subdomain decomposition
  # TOFIX: Remove indexes from Params
  params[:i_gm] = 1                                                  # subdomain decomposition
  params[:i_env] = 2                                                  # subdomain decomposition
  params[:i_uds] = (3,)                                                  # subdomain decomposition
  params[:i_sd] = (params[:i_env],params[:i_uds]...)                  # subdomain decomposition
  params[:i_sd] = (params[:i_env],params[:i_uds]...)                  # subdomain decomposition
  params[:n_updrafts] = length(params[:i_uds])                       # subdomain decomposition
  params[:entrainment_factor] = 1.0
  params[:detrainment_factor] = 1.0
  params[:tke_ed_coeff] = 0.1
  params[:prandtl_number] = 1.0
  params[:tke_diss_coeff] = 2.0
  params[:pressure_buoy_coeff] = 1.0/3.0
  params[:pressure_drag_coeff] = 0.375
  params[:pressure_plume_spacing] = 500.0

  params[:z_min] = 0.0                                                # spatial discretization
  params[:z_max] = 3000.0                                             # spatial discretization
  params[:N_elems] = 75                                              # spatial discretization
  params[:Δz] = (params[:z_max]-params[:z_min])/params[:N_elems]      # spatial discretization
  params[:Δt] = 20.0                                                    # temporal discretization
  params[:Δt_min] = 20.0                                                  #
  params[:t_end] = 21600.0                                       # temporal discretization
  params[:CFL] = 0.8                                                  #
  params[:Fo] = 0.5                                                   #
  params[:c_w] = 0.4                                                   # parameter in mixing length
  params[:c_ε] = 0.12                                                 # entr-detr factors
  params[:c_δ_0] = 0.12                                               # entr-detr factors
  params[:δ_B] = 0.004                                                # entr-detr factors
  params[:c_e] = 2.0                                                  # dissipation parameter
  params[:r_d] = 500                                                  # buoyancy factors
  params[:α_b] = 1/3                                                  # buoyancy factors
  params[:α_d] = 0.375                                                # buoyancy factors
  params[:f_coriolis] = 0.376e-4                                      # coriolis force coefficient
  params[:Prandtl_neutral] = 0.74                                     # Prandtl number
  params[:uni_func_a] = 4.7                                           #
  params[:Ψ_m_tol] = 0.0001                                           #
  params[:tol_abs] = 0.0001                                           #
  params[:iter_max] = 10                                              #
  params[:c_K] = 0.1                                                  # eddy-diffusivity coefficient
  params[:c_frac] = 0.1                                               # bc value for area fraction
  params[:a_surf] = 0.1                                               # area fraction at surface
  params[:surface_area] = params[:a_surf]                             # area fraction at surface
  params[:a_bounds] = [1e-3, 1-1e-3]                                  # filter for area fraction
  params[:w_bounds] = [0.0, 10000.0]                                  # filter for area fraction
  params[:q_bounds] = [0.0, 1.0]                                      # filter for area fraction
  params[:positivity_bounds] = [0, Inf]                                 # filter for area fraction
  params[:negativity_bounds] = [-Inf, 0]                                 # filter for area fraction

  params[:f_c] = 0                                                    # buoyancy gradient factor
  params[:zrough] = 1.0e-4                                            # surface roughness
  params[:ustar] = 0.28                                               # friction velocity?
  params[:Pg] = 1.015e5                                               # Reference state vars
  params[:Tg] = 300.4                                                 # Reference state vars
  params[:qtg] = 0.02245                                              # Reference state vars
  params[:ρg] = air_density(params[:Tg], params[:Pg], PhasePartition(params[:qtg]))
  params[:αg] = 1/params[:ρg]
  params[:Tsurface] = 299.1 * exner(params[:Pg])
  params[:qsurface] = 22.45e-3
  params[:lhf] = 5.2e-5 * params[:ρg] * latent_heat_vapor(params[:Tsurface])
  params[:shf] = 8.0e-3 * cp_m(PhasePartition(params[:qsurface])) * params[:ρg]
  params[:ρ_tflux] =  params[:shf] /(cp_m(PhasePartition(params[:qsurface])))
  params[:ρq_tot_flux] = params[:lhf]/(latent_heat_vapor(params[:Tsurface]))
  params[:ρθ_liq_flux] = params[:ρ_tflux] / exner(params[:Pg])
  params[:tke_surface_tol] = 0.01                                     # inversion height parameters

  params[:TCV_w_θ_liq] = 8*10^(-3)                                    # surface flux parameters
  params[:TCV_w_q_tot] = 5.2*10^(-5)                                  # surface flux parameters
  params[:inversion_height] = [1.0 for i in 1:params[:N_subdomains]]  # inversion height
  params[:Ri_bulk_crit] = 0.0                                         # inversion height parameters
  params[:bflux] = (grav * ((8.0e-3 + (molmass_ratio-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (molmass_ratio-1) * 22.45e-3))))
  params[:cq] = 0.001133                                              # Some surface parameter in SCAMPy
  params[:ch] = 0.001094                                              # Some surface parameter in SCAMPy
  params[:cm] = 0.001229                                              # Some surface parameter in SCAMPy
  params[:grid_adjust] = (log(20.0/params[:zrough])/log(params[:Δz]/2/params[:zrough]))^2 # Some surface parameter in SCAMPy
  params[:cq] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  params[:ch] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  params[:cm] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  params[:c_frac_i] = params[:c_frac]/(params[:N_subdomains]-1)
  params[:a_each] = vcat([i==params[:i_env] ? 1 - params[:c_frac] : params[:c_frac_i] for i in 1:params[:N_subdomains]]) # subdomain decomposition

  a_cumulative = vcat([0],cumsum(params[:a_each]))
  params[:surface_scalar_coeff] = [percentile_bounds_mean_norm(a_cumulative[i], a_cumulative[i+1], 1000) for i in 1:params[:N_subdomains]]

  params[:Δt] = Float64(params[:Δt])
  return params
end

"""
    Params(::Soares)

Initialize stand-alone input parameters to
solve the EDMF equations for the Soares case.
"""
function Params(::Soares)
  params = Dict()
  params[:N_subdomains] = 3                                           # subdomain decomposition
  # TOFIX: Remove indexes from Params
  params[:i_gm] = 1                                                  # subdomain decomposition
  params[:i_env] = 2                                                  # subdomain decomposition
  params[:i_uds] = (3,)                                                  # subdomain decomposition
  params[:i_sd] = (params[:i_env],params[:i_uds]...)                  # subdomain decomposition

  params[:z_min] = 0.0                                                # spatial discretization
  params[:z_max] = 3000.0                                             # spatial discretization
  params[:N_elems] = 150                                              # spatial discretization
  params[:Δz] = (params[:z_max]-params[:z_min])/params[:N_elems]      # spatial discretization
  # params[:N_elems] = 7                                              # spatial discretization
  # params[:Δt] = 20                                                    # temporal discretization
  # params[:t_end] = 21600.0                                             # temporal discretization
  # params[:Δt] = 0.01                                                     # temporal discretization
  params[:Δt] = 0.001                                                     # temporal discretization
  # params[:t_end] = 8000*params[:Δt]                                       # temporal discretization
  params[:t_end] = 8000.0                                            # temporal discretization
  # params[:t_end] = 10                                                # temporal discretization
  params[:CFL] = 0.5                                                  #
  params[:Fo] = 0.5                                                   #
  params[:Δt_min] = 10                                                  #
  # params[:t_end] = 100*3600*6*params[:Δt]                                       # temporal discretization
  # params[:t_end] = 4*params[:Δt]                                       # temporal discretization
  # params[:t_end] = 3000*params[:Δt]                                       # temporal discretization
  # params[:t_end] = 2*4*3600.0                                         # temporal discretization
  # params[:Δt] = 60.0                                                  # temporal discretization
  # params[:t_end] = 3600.0                                           # temporal discretization
  # params[:t_end] = 6*3600.0                                           # temporal discretization
  params[:c_w] = 0.4                                                   # parameter in mixing length
  params[:c_ε] = 0.004                                                 # entr-detr factors
  params[:c_δ_0] = 0.004                                               # entr-detr factors
  params[:δ_B] = 0.004                                                # entr-detr factors
  params[:c_e] = 0.0                                                  # dissipation parameter
  params[:r_d] = 500                                                  # buoyancy factors
  params[:TCV_w_θ_liq] = 6*10^(-2)                                    # surface flux conditions
  params[:TCV_w_q_tot] = 2.5*10^(-5)                                  # surface flux conditions
  # params[:TCV_w_θ_liq] = 6*10^(-3)                                    # surface flux conditions
  params[:α_b] = 1/3                                                  # buoyancy factors
  params[:α_d] = 0.375                                                # buoyancy factors
  params[:f_coriolis] = 0.376e-4                                      # coriolis force coefficient
  params[:uni_func_a] = 4.7                                           #
  params[:Ψ_m_tol] = 0.0001                                           #
  params[:tol_abs] = 0.0001                                           #
  params[:iter_max] = 10                                              #
  params[:c_K] = 0.1                                                  # eddy-diffusivity coefficient
  params[:c_frac] = 0.1                                               # bc value for area fraction
  params[:Prandtl_neutral] = 0.74                                     # Prandtl number
  params[:a_bounds] = [0.0001, 1-0.0001]                              # filter for area fraction
  params[:w_bounds] = [0.0, 10000.0]                                  # filter for area fraction
  params[:q_bounds] = [0.0, 1.0]                                      # filter for area fraction
  params[:positivity_bounds] = [0, Inf]                                 # filter for area fraction
  params[:negativity_bounds] = [-Inf, 0]                                 # filter for area fraction
  params[:Pg] = 1000.0 * 100.0                                        # Reference state vars
  params[:Tg] = 300.0                                                 # Reference state vars
  params[:f_c] = 0                                                    # buoyancy gradient factor
  params[:qtg] = 4.5e-3                                               # Reference state vars
  params[:qsurface] = 5e-3                                            # Reference state vars
  params[:zrough] = 1.0e-4                                            # surface roughness
  params[:ustar] = 0.28                                               # friction velocity?
  params[:inversion_height] = [1.0 for i in 1:params[:N_subdomains]]  # inversion height
  params[:Ri_bulk_crit] = 0.0                                         # inversion height parameters
  params[:tke_surface_tol] = 0.01                                     # inversion height parameters
  params[:bflux] = (grav * ((8.0e-3 + (molmass_ratio-1.0)*(299.1 * 5.2e-5  + 22.45e-3 * 8.0e-3)) /(299.1 * (1.0 + (molmass_ratio-1) * 22.45e-3))))
  params[:cq] = 0.001133                                              # Some surface parameter in SCAMPy
  params[:ch] = 0.001094                                              # Some surface parameter in SCAMPy
  params[:cm] = 0.001229                                              # Some surface parameter in SCAMPy
  # params[:grid_adjust] = (log(20.0/params[:zrough])/log(params[:Δz]/2/params[:zrough]))^2 # Some surface parameter in SCAMPy
  # params[:cq] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  # params[:ch] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  # params[:cm] *= params[:grid_adjust]                                 # Some surface parameter in SCAMPy
  params[:c_frac_i] = params[:c_frac]/(params[:N_subdomains]-1)


  params[:a_each] = vcat([i==params[:i_env] ? 1 - params[:c_frac] : (i==params[:i_gm] ? 1 : params[:c_frac_i]) for i in 1:params[:N_subdomains]]) # subdomain decomposition
  a_cumulative = vcat([0],cumsum([params[:a_each][i] for i in 1:params[:N_subdomains] if i != params[:i_gm]]))
  params[:surface_scalar_coeff] = [percentile_bounds_mean_norm(a_cumulative[i], a_cumulative[i+1], 1000) for i in 1:params[:N_subdomains]-1]
  params[:surface_scalar_coeff] = vcat([0],params[:surface_scalar_coeff])

  params[:Δt] = Float64(params[:Δt])

  # return NamedTuple{Tuple(keys(params))}(values(params))
  return params
end
