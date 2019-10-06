#### EDMF Functions

bound(x, x_bounds) = min(max(x, x_bounds[1]), x_bounds[2])

inside_bounds(x, x_bounds) = x > x_bounds[1] && x < x_bounds[2]

function assign_new_to_values!(grid, q_new, q, tmp)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems(grid), i in ud, name in (:w, :q_tot, :θ_liq)
    q_new[name, k, i] = q[name, k, i]
  end
end

function assign_values_to_new!(grid, q, q_new, tmp)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems(grid), i in ud
    @inbounds for name in (:a, :w, :q_tot, :θ_liq)
      q[name, k, i] = q_new[name, k, i]
    end
  end
  @inbounds for k in over_elems(grid)
    q[:tke  , k, en] = q_new[:tke, k, en]
    q[:a    , k, en] = 1 - sum([q_new[:a, k, i] for i in ud])
    q[:θ_liq, k, gm] = q_new[:θ_liq, k, gm]
    q[:q_tot, k, gm] = q_new[:q_tot, k, gm]
  end
end

function compute_new_ud_a!(grid, q_new, q, q_tendencies, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      a_predict = q[:a, k, i] + params[:Δt][1] * q_tendencies[:a, k, i]
      q_new[:a, k, i] = bound(a_predict, params[:a_bounds])
    end
  end
end

function compute_tendencies_en_O2!(grid::Grid{FT}, q_tendencies, tmp_O2, cv) where FT
  gm, en, ud, sd, al = allcombinations(q_tendencies)
  k_1 = first_interior(grid, Zmin())
  @inbounds for k in over_elems_real(grid)
    q_tendencies[cv, k, en] = tmp_O2[cv][:press, k] +
                              tmp_O2[cv][:buoy, k] +
                              tmp_O2[cv][:shear, k] +
                              tmp_O2[cv][:entr_gain, k] +
                              tmp_O2[cv][:rain_src, k]
  end
  q_tendencies[cv, k_1, en] = FT(0)
end

function compute_tendencies_gm_scalars!(grid, q_tendencies, q, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  k_1 = first_interior(grid, Zmin())
  Δzi = grid.Δzi
  α_1 = tmp[:α_0, k_1]
  ae_1 = q[:a, k_1, en]
  @inbounds for k in over_elems(grid)
    q_tendencies[:q_tot, k, gm] += tmp[:mf_tend_q_tot, k]
    q_tendencies[:θ_liq, k, gm] += tmp[:mf_tend_θ_liq, k]
  end

  q_tendencies[:q_tot, k_1, gm] += params[:ρq_tot_flux] * Δzi * α_1/ae_1
  q_tendencies[:θ_liq, k_1, gm] += params[:ρθ_liq_flux] * Δzi * α_1/ae_1
end

function compute_tendencies_ud!(grid, q_tendencies, q, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)

      a_k = q[:a, k, i]
      α_0_kp = tmp[:α_0, k]
      w_env = q[:w, k, en]
      ρ_k = tmp[:ρ_0, k]
      w_i = q[:w, k, i]
      ε_model = tmp[:ε_model, k, i]
      δ_model = tmp[:δ_model, k, i]
      θ_liq_env = q[:θ_liq, k, en]
      q_tot_env = q[:q_tot, k, en]
      B_k = tmp[:buoy, k, i]
      ρa_k = ρ_k * a_k
      ρaw_k = ρa_k * w_i

      a_cut     = q[:a, Cut(k), i]
      θ_liq_cut = q[:θ_liq, Cut(k), i]
      q_tot_cut = q[:q_tot, Cut(k), i]
      w_cut     = q[:w, Cut(k), i]
      ρ_cut     = tmp[:ρ_0, Cut(k)]

      ρaw_cut = ρ_cut .* a_cut .* w_cut
      ρawθ_liq_cut = ρaw_cut .* θ_liq_cut
      ρawq_tot_cut = ρaw_cut .* q_tot_cut
      ρaww_cut = ρ_cut .* a_cut .* w_cut .* w_cut

      tendencies = 0
      adv = - α_0_kp * advect(ρaw_cut, w_cut, grid)
      tendencies+=adv
      ε_term =   a_k * w_i * ε_model
      tendencies+=ε_term
      δ_term = - a_k * w_i * δ_model
      tendencies+=δ_term
      q_tendencies[:a, k, i] = tendencies

      adv = -advect(ρaww_cut, w_cut, grid)
      exch = ρaw_k * (- δ_model * w_i + ε_model * w_env)
      buoy = ρa_k * B_k
      press_buoy = - ρa_k * B_k * params[:pressure_buoy_coeff]
      p_coeff = params[:pressure_drag_coeff]/params[:pressure_plume_spacing]
      press_drag = - ρa_k * (p_coeff * (w_i - w_env)^2/sqrt(a_k))
      nh_press = press_buoy + press_drag

      tendencies = (adv + exch + buoy + nh_press)
      q_tendencies[:w, k, i] = tendencies

      tendencies_θ_liq = 0
      tendencies_q_tot = 0

      tendencies_θ_liq += -advect(ρawθ_liq_cut, w_cut, grid)
      tendencies_q_tot += -advect(ρawq_tot_cut, w_cut, grid)

      tendencies_θ_liq += ρaw_k * (ε_model * θ_liq_env - δ_model * θ_liq_cut[2])
      tendencies_q_tot += ρaw_k * (ε_model * q_tot_env - δ_model * q_tot_cut[2])

      q_tendencies[:θ_liq, k, i] = tendencies_θ_liq
      q_tendencies[:q_tot, k, i] = tendencies_q_tot
    end
  end
end


function compute_new_ud_w!(grid, q_new, q, q_tendencies, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  # Solve for updraft velocity
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      a_new_k = q_new[:a, k, i]
      ρ_k     = tmp[:ρ_0, k]
      w_i     = q[:w, k, i]
      a_k     = q[:a, k, i]
      ρa_k = ρ_k * a_k
      ρa_new_k = ρ_k * a_new_k
      ρaw_k = ρa_k * w_i
      w_predict = ρaw_k/ρa_new_k + params[:Δt][1]/ρa_new_k*q_tendencies[:w, k, i]
      q_new[:w, k, i] = bound(w_predict, params[:w_bounds])
    end
  end
end

function compute_new_ud_scalars!(grid, q_new, q, q_tendencies, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      a_k = q[:a, k, i]
      a_k_new = q_new[:a, k, i]
      ρ_k = tmp[:ρ_0, k]
      ρa_k = ρ_k*a_k
      ρa_new_k = ρ_k * a_k_new
      θ_liq_predict = (ρa_k * q[:θ_liq, k, i] + params[:Δt][1]*q_tendencies[:θ_liq, k, i])/ρa_new_k
      q_tot_predict = (ρa_k * q[:q_tot, k, i] + params[:Δt][1]*q_tendencies[:q_tot, k, i])/ρa_new_k
      q_new[:θ_liq, k, i] = θ_liq_predict
      q_new[:q_tot, k, i] = q_tot_predict
    end
  end
end

function compute_new_en_O2!(grid, q_new, q, q_tendencies, tmp, tmp_O2, params, cv, tri_diag)
  gm, en, ud, sd, al = allcombinations(q)
  construct_tridiag_diffusion_O2!(grid, q, tmp, params, tri_diag)
  k_1 = first_interior(grid, Zmin())
  @inbounds for k in over_elems(grid)
    tri_diag[:f, k] = tmp[:ρ_0, k] * q[:a, k, en] * q[cv, k, en] * (1/params[:Δt][1]) + q_tendencies[cv, k, en]
  end
  tri_diag[:f, k_1] = tmp[:ρ_0, k_1] * q[:a, k_1, en] * q[cv, k_1, en] * (1/params[:Δt][1]) + q[cv, k_1, en]
  solve_tridiag_wrapper!(grid, q_new, cv, en, tri_diag)
end


function compute_new_gm_scalars!(grid, q_new, q, q_tendencies, params, tmp, tri_diag)
  gm, en, ud, sd, al = allcombinations(q)

  @inbounds for k in over_elems_real(grid)
    tri_diag[:ρaK, k] = q[:a, k, en]*tmp[:K_h, k, gm]*tmp[:ρ_0, k]
  end
  construct_tridiag_diffusion_O1!(grid, q, tmp, params[:Δt][1], tri_diag)

  @inbounds for k in over_elems(grid)
    tri_diag[:f, k] = q[:q_tot, k, gm] + params[:Δt][1]*q_tendencies[:q_tot, k, gm]
  end
  solve_tridiag_wrapper!(grid, q_new, :q_tot, gm, tri_diag)

  @inbounds for k in over_elems(grid)
    tri_diag[:f, k] = q[:θ_liq, k, gm] + params[:Δt][1]*q_tendencies[:θ_liq, k, gm]
  end
  solve_tridiag_wrapper!(grid, q_new, :θ_liq, gm, tri_diag)
end


function saturation_adjustment_sd!(grid, q, tmp)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for i in sd
    @inbounds for k in over_elems_real(grid)
      ts = ActiveThermoState(q, tmp, k, i)
      q_liq = PhasePartition(ts).liq
      T = air_temperature(ts)
      tmp[:T, k, i] = T
      tmp[:q_liq, k, i] = q_liq
    end
  end
end

abstract type EntrDetrModel end

struct BOverW2 <: EntrDetrModel end
function compute_entrainment_detrainment!(grid::Grid{FT}, UpdVar, tmp, q, params, ::BOverW2) where FT
  gm, en, ud, sd, al = allcombinations(q)
  Δzi = grid.Δzi
  k_1 = first_interior(grid, Zmin())
  @inbounds for i in ud
    zi = UpdVar[i].cloud.base
    @inbounds for k in over_elems_real(grid)
      buoy = tmp[:buoy, k, i]
      w = q[:w, k, i]
      if grid.zc[k] >= zi
        detr_sc = 4.0e-3 + 0.12 *abs(min(buoy,0.0)) / max(w * w, 1e-2)
      else
        detr_sc = FT(0)
      end
      entr_sc = 0.12 * max(buoy, FT(0) ) / max(w * w, 1e-2)
      tmp[:ε_model, k, i] = entr_sc * params[:entrainment_factor]
      tmp[:δ_model, k, i] = detr_sc * params[:detrainment_factor]
    end
    tmp[:ε_model, k_1, i] = 2 * Δzi
    tmp[:δ_model, k_1, i] = FT(0)
  end
end

function compute_cloud_phys!(grid::Grid{FT}, q, tmp) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    q_tot = q[:q_tot, k, en]
    ts = ActiveThermoState(q, tmp, k, en)
    T = air_temperature(ts)
    q_liq = PhasePartition(ts).liq
    q_vap = q_tot - q_liq
    θ = dry_pottemp(ts)
    if q_liq > 0
      tmp[:CF, k] = FT(1)
      tmp[:θ_cloudy, k]     = θ
      tmp[:t_cloudy, k]     = T
      tmp[:q_tot_cloudy, k] = q_tot
      tmp[:q_vap_cloudy, k] = q_vap
    else
      tmp[:CF, k] = FT(0)
      tmp[:θ_dry, k]     = θ
      tmp[:q_tot_dry, k] = q_tot
    end
  end
end

@inline buoyancy(α_0, α) = grav * (α - α_0)/α_0

function compute_buoyancy!(grid, q, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for i in (ud...,en)
    @inbounds for k in over_elems_real(grid)
      q_tot = q[:q_tot, k, i]
      q_liq = tmp[:q_liq, k, i]
      T = tmp[:T, k, i]
      α_i = specific_volume(T, tmp[:p_0, k], PhasePartition(q_tot, q_liq))
      tmp[:buoy, k, i] = buoyancy(tmp[:α_0, k], α_i)
    end
  end

  # Filter buoyancy
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      weight = tmp[:HVSD_a, k, i]
      tmp[:buoy, k, i] = weight*tmp[:buoy, k, i] + (1-weight)*tmp[:buoy, k, en]
    end
  end

  # Subtract grid mean buoyancy
  @inbounds for k in over_elems_real(grid)
    tmp[:buoy, k, gm] = sum([q[:a, k, i] * tmp[:buoy, k, i] for i in sd])
    @inbounds for i in sd
      tmp[:buoy, k, i] -= tmp[:buoy, k, gm]
    end
  end
end

function top_of_updraft(grid::Grid, q::StateVec, params)
  @unpack params w_bounds a_bounds
  gm, en, ud, sd, al = allcombinations(q)

  z_star_a = zeros(length(ud))
  z_star_w = zeros(length(ud))
  k_2 = first_interior(grid, Zmax())
  @inbounds for i in ud
    z_star_a[i] = min([!inside_bounds(q[:a, k, i], a_bounds) ? grid.zc[k] : grid.zc[k_2+1] for k in over_elems_real(grid)]...)
    z_star_w[i] = min([!inside_bounds(q[:w, k, i], w_bounds) ? grid.zc[k] : grid.zc[k_2+1] for k in over_elems_real(grid)]...)
  end
  return z_star_a, z_star_w
end

function filter_scalars!(grid, q, tmp, params)
  gm, en, ud, sd, al = allcombinations(q)
  z_star_a, z_star_w = top_of_updraft(grid, q, params)
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      tmp[:HVSD_a, k, i] = 1 - heaviside(grid.zc[k] - z_star_a[i], 1)
      tmp[:HVSD_w, k, i] = 1 - heaviside(grid.zc[k] - z_star_w[i], 1)
    end
  end

  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)[2:end]
      q[:w, k, i] = bound(q[:w, k, i]*tmp[:HVSD_w, k, i], params[:w_bounds])
      q[:a, k, i] = bound(q[:a, k, i]*tmp[:HVSD_w, k, i], params[:a_bounds])

      weight = tmp[:HVSD_w, k, i]
      q[:θ_liq, k, i] = weight*q[:θ_liq, k, i] + (1-weight)*q[:θ_liq, k, gm]
      q[:q_tot, k, i] = weight*q[:q_tot, k, i] + (1-weight)*q[:q_tot, k, gm]
    end
  end
end

function compute_cv_gm!(grid, q, ϕ, ψ, cv, tke_factor)
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems(grid)
    Δϕ = q[ϕ, k, en] - q[ϕ, k, gm]
    Δψ = q[ψ, k, en] - q[ψ, k, gm]
    q[cv, k, en] = tke_factor * q[:a, k, en] * Δϕ * Δψ + q[:a, k, en] * q[cv, k, en]
    @inbounds for i in ud
      Δϕ = q[ϕ, k, i] - q[ϕ, k, gm]
      Δψ = q[ψ, k, i] - q[ψ, k, gm]
      q[cv, k, en] += tke_factor * q[:a, k, i] * Δϕ * Δψ
    end
  end
end

function compute_mf_gm!(grid, q, tmp)
  gm, en, ud, sd, al = allcombinations(q)
  domain_c = over_elems_real(grid)

  @inbounds for i in ud
    @inbounds for k in domain_c
      tmp[:mf_tmp, k, i] = (q[:w, k, i] - q[:w, k, en]) * tmp[:ρ_0, k] * q[:a, k, i]
    end
  end

  @inbounds for k in domain_c
    tmp[:mf_θ_liq, k] = sum([tmp[:mf_tmp, k, i] * (q[:θ_liq, k, i] - q[:θ_liq, k, en]) for i in ud])
    tmp[:mf_q_tot, k] = sum([tmp[:mf_tmp, k, i] * (q[:q_tot, k, i] - q[:q_tot, k, en]) for i in ud])
  end

  @inbounds for k in domain_c
    tmp[:mf_tend_θ_liq, k] = -tmp[:α_0, k]*grad(tmp[:mf_θ_liq, Cut(k)], grid)
    tmp[:mf_tend_q_tot, k] = -tmp[:α_0, k]*grad(tmp[:mf_q_tot, Cut(k)], grid)
  end
end

abstract type MixingLengthModel end
struct ConstantMixingLength{FT} <: MixingLengthModel
  value::FT
end
function compute_mixing_length!(grid::Grid{FT}, q, tmp, params, model::ConstantMixingLength) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems(grid)
    tmp[:l_mix, k, gm] = model.value
  end
end

function compute_eddy_diffusivities_tke!(grid::Grid{FT}, q, tmp, params) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    l_mix = tmp[:l_mix, k, gm]
    K_m_k = params[:tke_ed_coeff] * l_mix * sqrt(max(q[:tke, k, en], FT(0)))
    tmp[:K_m, k, gm] = K_m_k
    tmp[:K_h, k, gm] = K_m_k / params[:prandtl_number]
  end
end

function compute_tke_buoy!(grid::Grid{FT}, q, tmp, tmp_O2, cv) where FT
  gm, en, ud, sd, al = allcombinations(q)

  # Note that source terms at the first interior point are not really used because that is where tke boundary condition is
  # enforced (according to MO similarity). Thus here I am being sloppy about lowest grid point
  @inbounds for k in over_elems_real(grid)
    q_tot_dry    = tmp[:q_tot_dry, k]
    θ_dry        = tmp[:θ_dry, k]
    t_cloudy     = tmp[:t_cloudy, k]
    q_vap_cloudy = tmp[:q_vap_cloudy, k]
    q_tot_cloudy = tmp[:q_tot_cloudy, k]
    θ_cloudy     = tmp[:θ_cloudy, k]
    p_0          = tmp[:p_0, k]

    lh = latent_heat_vapor(t_cloudy)
    cpm = cp_m(PhasePartition(q_tot_cloudy))
    grad_θ_liq = ∇_neg(q[:θ_liq, Cut(k), en], grid)
    grad_q_tot = ∇_neg(q[:q_tot, Cut(k), en], grid)

    prefactor = R_d * exner(p_0)/p_0
    ε_vi = R_v / R_d

    d_alpha_θ_liq_dry = prefactor * (1 + (ε_vi - 1) * q_tot_dry)
    d_alpha_q_tot_dry = prefactor * θ_dry * (ε_vi - 1)
    CF_env = tmp[:CF, k]

    if CF_env > 0
      d_alpha_θ_liq_cloudy = (prefactor * (1 + ε_vi * (1 + lh / R_v / t_cloudy) * q_vap_cloudy - q_tot_cloudy )
                               / (1 + lh * lh / cpm / R_v / t_cloudy / t_cloudy * q_vap_cloudy))
      d_alpha_q_tot_cloudy = (lh / cpm / t_cloudy * d_alpha_θ_liq_cloudy - prefactor) * θ_cloudy
    else
      d_alpha_θ_liq_cloudy = 0
      d_alpha_q_tot_cloudy = 0
    end

    d_alpha_θ_liq_total = (CF_env * d_alpha_θ_liq_cloudy + (1-CF_env) * d_alpha_θ_liq_dry)
    d_alpha_q_tot_total = (CF_env * d_alpha_q_tot_cloudy + (1-CF_env) * d_alpha_q_tot_dry)

    K_h_k = tmp[:K_h, k, gm]
    term_1 = - K_h_k * grad_θ_liq * d_alpha_θ_liq_total
    term_2 = - K_h_k * grad_q_tot * d_alpha_q_tot_total

    # TODO - check
    tmp_O2[cv][:buoy, k] = grav / tmp[:α_0, k] * q[:a, k, en] * tmp[:ρ_0, k] * (term_1 + term_2)
  end
end


function compute_cv_entr!(grid::Grid{FT}, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    tmp_O2[cv][:entr_gain, k] = FT(0)
    @inbounds for i in ud
      Δϕ = q[ϕ, k, i] - q[ϕ, k, en]
      Δψ = q[ψ, k, i] - q[ψ, k, en]
      tmp_O2[cv][:entr_gain, k] += tke_factor*q[:a, k, i] * abs(q[:w, k, i]) * tmp[:δ_model, k, i] * Δϕ * Δψ
    end
    tmp_O2[cv][:entr_gain, k] *= tmp[:ρ_0, k]
  end
end

function compute_cv_shear!(grid::Grid{FT}, q, tmp, tmp_O2, ϕ, ψ, cv) where FT
  gm, en, ud, sd, al = allcombinations(q)
  is_tke = cv==:tke
  tke_factor = is_tke ? FT(0.5) : 1
  grad_u = 0
  grad_v = 0
  @inbounds for k in over_elems_real(grid)
    if is_tke
      grad_u = ∇_neg(q[:u, Cut(k), gm], grid)
      grad_v = ∇_neg(q[:v, Cut(k), gm], grid)
      grad_ϕ = ∇_neg(q[ϕ, Cut(k), en], grid)
      grad_ψ = ∇_neg(q[ψ, Cut(k), en], grid)
    else
      grad_ϕ = grad(q[ϕ, Cut(k), en], grid)
      grad_ψ = grad(q[ψ, Cut(k), en], grid)
    end
    ρaK = tmp[:ρ_0, k] * q[:a, k, en] * tmp[:K_h, k, gm]
    tmp_O2[cv][:shear, k] = tke_factor*2*ρaK * (grad_ϕ*grad_ψ + grad_u^2 + grad_v^2)
  end
end

function compute_cv_interdomain_src!(grid::Grid{FT}, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor) where FT
    gm, en, ud, sd, al = allcombinations(q)
    @inbounds for k in over_elems(grid)
        tmp_O2[cv][:interdomain, k] = FT(0)
        @inbounds for i in ud
            Δϕ = q[ϕ, k, i] - q[ϕ, k, en]
            Δψ = q[ψ, k, i] - q[ψ, k, en]
            tmp_O2[cv][:interdomain, k] += tke_factor*q[:a, k, i] * (1-q[:a, k, i]) * Δϕ * Δψ
        end
    end
end

function compute_tke_pressure!(grid::Grid{FT}, q, tmp, tmp_O2, cv, params) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    tmp_O2[cv][:press, k] = FT(0)
    @inbounds for i in ud
      wu_half = q[:w, k, i]
      we_half = q[:w, k, en]
      a_i = q[:a, k, i]
      ρ_0_k = tmp[:ρ_0, k]
      press_buoy = (-1 * ρ_0_k * a_i * tmp[:buoy, k, i] * params[:pressure_buoy_coeff])
      press_drag_coeff = -1 * ρ_0_k * sqrt(a_i) * params[:pressure_drag_coeff]/params[:pressure_plume_spacing]
      press_drag = press_drag_coeff * (wu_half - we_half)*abs(wu_half - we_half)
      tmp_O2[cv][:press, k] += (we_half - wu_half) * (press_buoy + press_drag)
    end
  end
end

function compute_cv_env!(grid::Grid{FT}, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems(grid)
    if q[:a, k, en] > 0
      @inbounds for i in sd
        Δϕ = q[ϕ, k, i] - q[ϕ, k, gm]
        Δψ = q[ψ, k, i] - q[ψ, k, gm]
        q[cv, k, en] -= tke_factor * q[:a, k, i] * Δϕ * Δψ
      end
      q[cv, k, en] = q[cv, k, en]/q[:a, k, en]
    else
      q[cv, k, en] = FT(0)
    end
  end
end

function cleanup_covariance!(grid::Grid{FT}, q) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    q[:tke, k, en] = q[:tke, k, en] < eps(FT) ? FT(0) : q[:tke, k, en]
  end
end


function construct_tridiag_diffusion_O1!(grid::Grid{FT}, q, tmp, Δt, tri_diag) where FT
  gm, en, ud, sd, al = allcombinations(tmp)
  k_1 = first_interior(grid, Zmin())
  k_2 = first_interior(grid, Zmax())
  dzi = grid.Δzi
  @inbounds for k in over_elems_real(grid)
    ρaK_cut = tri_diag[:ρaK, Cut(k)]
    X = tmp[:ρ_0, k] * q[:a, k, en]/Δt
    Z = ρaK_cut[1] * dzi * dzi
    Y = ρaK_cut[2] * dzi * dzi
    if k == k_1
      Z = FT(0)
    elseif k == k_2
      Y = FT(0)
    end
    tri_diag[:a, k] = - Z/X
    tri_diag[:b, k] = 1 + Y/X + Z/X
    tri_diag[:c, k] = -Y/X
  end
end

function construct_tridiag_diffusion_O2!(grid::Grid{FT}, q, tmp, params, tri_diag) where FT
  gm, en, ud, sd, al = allcombinations(q)
  Δzi = grid.Δzi
  Δzi2 = grid.Δzi2
  dti = 1/params[:Δt][1]
  k_1 = first_interior(grid, Zmin())
  k_2 = first_interior(grid, Zmax())

  @inbounds for k in over_elems_real(grid)
    ρ_0_cut = tmp[:ρ_0, Cut(k)]
    ae_cut = q[:a, Cut(k), en]
    w_cut = q[:w, Cut(k), en]
    ρa_K = q[:a, Cut(k), en] .* tmp[:K_h, Cut(k), gm] .* tmp[:ρ_0, Cut(k)]

    D_env = sum([ρ_0_cut[2] *
                 q[:a, k, i] *
                 q[:w, k, i] *
                 tmp[:ε_model, k, i] for i in ud])

    l_mix = max(tmp[:l_mix, k, gm], FT(1))
    tke_env = max(q[:tke, k, en], FT(0))

    tri_diag[:a, k] = (- ρa_K[2] * Δzi2 )
    tri_diag[:b, k] = (ρ_0_cut[2] * ae_cut[2] * dti
                     - ρ_0_cut[2] * ae_cut[2] * w_cut[2] * Δzi
                     + ρa_K[3] * Δzi2 + ρa_K[2] * Δzi2
                     + D_env
                     + ρ_0_cut[2] * ae_cut[2] * params[:tke_diss_coeff] * sqrt(tke_env)/l_mix)
    tri_diag[:c, k] = (ρ_0_cut[3] * ae_cut[3] * w_cut[3] * Δzi - ρa_K[3] * Δzi2)
  end

  tri_diag[:a, k_1] = FT(0)
  tri_diag[:b, k_1] = FT(1)
  tri_diag[:c, k_1] = FT(0)

  tri_diag[:b, k_2] += tri_diag[:c, k_2]
  tri_diag[:c, k_2] = FT(0)
end
