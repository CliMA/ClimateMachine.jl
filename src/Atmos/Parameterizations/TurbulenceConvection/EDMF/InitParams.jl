#### InitParams

export Params

"""
    Params(::Case)

Initialize stand-alone input parameters to
solve the EDMF equations for a given case.
"""
function Params end

function Params(::BOMEX)
  FT = Float64
  params = Dict()
  params[:export_data] = false
  params[:plot_single_fields] = true
  params[:export_frequency] = 2000
  params[:EntrDetrModel] = BOverW2()
  params[:MixingLengthModel] = ConstantMixingLength(FT(100))

  params[:N_subdomains] = 3
  # TOFIX: Remove indexes from Params
  params[:entrainment_factor]     = FT(1.0)
  params[:detrainment_factor]     = FT(1.0)
  params[:tke_ed_coeff]           = FT(0.1)
  params[:prandtl_number]         = FT(1.0)
  params[:tke_diss_coeff]         = FT(2.0)
  params[:pressure_buoy_coeff]    = FT(1.0/3.0)
  params[:pressure_drag_coeff]    = FT(0.375)
  params[:pressure_plume_spacing] = FT(500.0)

  params[:z_min] = 0.0
  params[:z_max] = 3000.0
  params[:N_elems] = 75
  params[:Δz] = (params[:z_max]-params[:z_min])/params[:N_elems]
  params[:Δt] = 20.0
  params[:Δt_min] = 20.0
  params[:t_end] = 21600.0
  params[:CFL] = 0.8                                                  #
  params[:f_coriolis] = 0.376e-4                                      # coriolis force coefficient
  params[:surface_area] = 0.1                                         # area fraction at surface
  params[:a_bounds] = [1e-3, 1-1e-3]                                  # filter for a
  params[:w_bounds] = [0.0, 10000.0]                                  # filter for w
  params[:q_bounds] = [0.0, 1.0]                                      # filter for q

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

  params[:Δt] = FT(params[:Δt])
  return params
end
