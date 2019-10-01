#### ApplyBCs

export apply_bcs!

function apply_bcs!(grid::Grid, q::StateVec, tmp::StateVec, params, case::Case) end

function apply_bcs!(grid::Grid, q::StateVec, tmp::StateVec, params, case::BOMEX)
  @unpack params ρq_tot_flux ρθ_liq_flux obukhov_length ustar n_updrafts surface_area wstar UpdVar
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  k_1 = first_interior(grid, Zmin())
  zLL = grid.zc[k_1]
  θ_liq_1 = q[:θ_liq, k_1, gm]
  q_tot_1 = q[:q_tot, k_1, gm]
  alpha0LL  = tmp[:α_0, k_1]
  cv_q_tot = surface_variance(ρq_tot_flux*alpha0LL, ρq_tot_flux*alpha0LL, ustar, zLL, obukhov_length)
  cv_θ_liq = surface_variance(ρθ_liq_flux*alpha0LL, ρθ_liq_flux*alpha0LL, ustar, zLL, obukhov_length)
  @inbounds for i in ud
    UpdVar[i].surface_bc.a = surface_area/n_updrafts
    UpdVar[i].surface_bc.w = 0.0
    UpdVar[i].surface_bc.θ_liq = (θ_liq_1 + UpdVar[i].surface_scalar_coeff * sqrt(cv_θ_liq))
    UpdVar[i].surface_bc.q_tot = (q_tot_1 + UpdVar[i].surface_scalar_coeff * sqrt(cv_q_tot))
    q[:a, k_1, i] = UpdVar[i].surface_bc.a
    q[:θ_liq, k_1, i] = UpdVar[i].surface_bc.θ_liq
    q[:q_tot, k_1, i] = UpdVar[i].surface_bc.q_tot
  end
  @inbounds for i in sd
    apply_Neumann!(q, :θ_liq, grid, 0.0, (Zmin(),Zmax()), i)
    apply_Neumann!(q, :q_tot, grid, 0.0, (Zmin(),Zmax()), i)
  end
  apply_Dirichlet!(q, :w  , grid, 0.0, (Zmin(),Zmax()), en)
  apply_Neumann!(q, :tke, grid, 0.0, (Zmin(),Zmax()), en)
  q[:tke, k_1, en] = surface_tke(ustar, wstar, zLL, obukhov_length)
end
