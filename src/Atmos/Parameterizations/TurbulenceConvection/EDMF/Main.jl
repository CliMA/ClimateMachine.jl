#### Main

"""
    run(case)
Solve the Eddy-Diffusivity Mass-Flux (EDMF) equations for a
stand-alone `case`
"""
function run(case)
  params = Params(case)

  tc = TurbConv(params, case)

  grid = tc.grid
  q = tc.q
  q_new = tc.q_new
  tmp = tc.tmp
  q_tendencies = tc.tendencies
  dir_tree = tc.dir_tree
  tri_diag = tc.tri_diag
  tmp_O2 = tc.tmp_O2

  gm, en, ud, sd, al = allcombinations(DomainIdx(tc.q))
  init_ref_state!(tmp, grid, params, dir_tree)
  init_state_vecs!(q, tmp, grid, params, dir_tree, case)


  params[:UpdVar] = [UpdraftVar(0, params[:a_surf], length(ud)) for i in al]
  # export_initial_conditions(q, tmp, grid, dir_tree[:processed_initial_conditions], true)

  @unpack params Δt t_end

  i_Δt, i_export, t = [0], [0], [0.0]

  assign!(q_tendencies, (:u, :v, :q_tot, :θ_liq), grid, 0.0)
  update_surface!(tmp, q, grid, params, case)
  update_forcing!(tmp, q, grid, params, case)
  compute_cloud_base_top_cover!(params[:UpdVar], grid, q, tmp)

  pre_compute_vars!(grid, q, tmp, tmp_O2, params[:UpdVar], params)

  apply_bcs!(grid, q, tmp, params, case)

  while t[1] < t_end
    assign!(q_tendencies, (:u, :v, :q_tot, :θ_liq), grid, 0.0)

    update_surface!(tmp, q, grid, params, case)
    update_forcing!(tmp, q, grid, params, case)

    pre_compute_vars!(grid, q, tmp, tmp_O2, params[:UpdVar], params)

    update!(grid, q_new, q, q_tendencies, tmp, tmp_O2, case, tri_diag, params)

    update_dt!(grid, params, q, t)

    domain_average!(tmp, q, :T, :a, grid)
    domain_average!(tmp, q, :q_liq, :a, grid)
    domain_average!(tmp, q, :buoy, :a, grid)
    compute_cloud_base_top_cover!(params[:UpdVar], grid, q, tmp)

    export_unsteady(t, i_Δt, i_export, params, q, tmp, grid, dir_tree)
  end

  export_plots(q, tmp, grid, dir_tree[:solution_raw]*"LastTimeStep", true, params, i_Δt)
  export_plots(q, tmp, grid, dir_tree[:solution_processed]*"LastTimeStep", false, params, i_Δt)

  export_data(q, tmp, grid, dir_tree)
  return (grid, q, tmp)
end

