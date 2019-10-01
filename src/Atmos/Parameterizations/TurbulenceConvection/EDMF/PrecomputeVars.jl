#### PrecomputeVars

abstract type BuoyancyModel end
struct BOverW2 <: BuoyancyModel end

function pre_compute_vars!(grid, q, tmp, tmp_O2, UpdVar, params)
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))

  diagnose_environment!(q, grid, :a, (:q_tot, :θ_liq, :w))

  saturation_adjustment_sd!(grid, q, tmp)

  @inbounds for k in over_elems_real(grid)
    ts = ActiveThermoState(q, tmp, k, gm)
    tmp[:θ_ρ, k] = virtual_pottemp(ts)
  end
  params[:zi] = compute_inversion_height(tmp, q, grid, params)
  params[:wstar] = compute_convective_velocity(params[:bflux], params[:zi])

  compute_entrainment_detrainment!(grid, UpdVar, tmp, q, params, BOverW2())
  compute_cloud_phys!(grid, q, tmp)
  compute_buoyancy!(grid, q, tmp, params)

  filter_scalars!(grid, q, tmp, params)

  compute_cv_gm!(grid, q, :w, :w, :tke, 0.5)
  compute_mf_gm!(grid, q, tmp)
  compute_mixing_length!(grid, q, tmp, params)
  compute_eddy_diffusivities_tke!(grid, q, tmp, params)

  compute_tke_buoy!(grid, q, tmp, tmp_O2, :tke)
  compute_cv_entr!(grid, q, tmp, tmp_O2, :w, :w, :tke, 0.5)
  compute_cv_shear!(grid, q, tmp, tmp_O2, :w, :w, :tke)
  compute_cv_interdomain_src!(grid, q, tmp, tmp_O2, :w, :w, :tke, 0.5)
  compute_tke_pressure!(grid, q, tmp, tmp_O2, :tke, params)
  compute_cv_env!(grid, q, tmp, tmp_O2, :w, :w, :tke, 0.5)

  cleanup_covariance!(grid, q)

end

