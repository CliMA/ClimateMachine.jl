#### InitialConditions

using DifferentialEquations

"""
    init_state_vecs!

Defines initial conditions for state vectors `q` and `tmp` for all sub-domains.
"""
function init_state_vecs! end

"""
    initialize_updrafts!

Defines initial conditions for state vectors `q` and `tmp` for the updraft sub-domains.
"""
function initialize_updrafts! end

"""
    init_forcing!

Defines initial conditions for forcing terms in the state vector `tmp` for all sub-domains.
"""
function init_forcing! end


function initialize_updrafts!(q::StateVec, tmp::StateVec, grid::Grid, params, dir_tree::DirTree, ::BOMEX)
  gm, en, ud, sd, al = allcombinations(q)
  k_1 = first_interior(grid, Zmin())
  n_updrafts = length(ud)
  for i in ud
    for k in over_elems(grid)
      q[:w, k, i] = 0
      q[:a, k, i] = bound(0.0, params[:a_bounds])
    end
    q[:a, k_1, i] = bound(params[:surface_area]/n_updrafts, params[:a_bounds])
  end
  return
end

function init_state_vecs!(q::StateVec, tmp::StateVec, grid::Grid, params, dir_tree::DirTree, case::BOMEX)
  @unpack params qtg Tg Pg a_bounds surface_area
  z = grid.zc

  gm, en, ud, sd, al = allcombinations(q)
  k_1 = first_interior(grid, Zmin())

  @inbounds for k in over_elems(grid)

    @inbounds for ϕ in var_names(q)
      @inbounds for i in over_sub_domains(q, ϕ)
        q[ϕ, k, i] = 0.0
      end
    end

    q[:a, k, gm] = 1.0
    for i in ud
      q[:a, k, i] = bound(0.0, a_bounds)
      k==k_1 && (q[:a, k, i] = bound(surface_area, a_bounds))
    end
    q[:a, k, en] = q[:a, k, gm] - sum([q[:a, k, i] for i in ud])

    # Set qt profile
    if z[k] <= 520
      q[:q_tot, k, gm] = (17.0 + (z[k]) * (16.3-17.0)/520.0)/1000.0
    end
    if z[k] > 520.0 && z[k] <= 1480.0
      q[:q_tot, k, gm] = (16.3 + (z[k] - 520.0)*(10.7 - 16.3)/(1480.0 - 520.0))/1000.0
    end
    if z[k] > 1480.0 && z[k] <= 2000.0
      q[:q_tot, k, gm] = (10.7 + (z[k] - 1480.0) * (4.2 - 10.7)/(2000.0 - 1480.0))/1000.0
    end
    if z[k] > 2000.0
      q[:q_tot, k, gm] = (4.2 + (z[k] - 2000.0) * (3.0 - 4.2)/(3000.0  - 2000.0))/1000.0
    end

    # Set θ_liq profile
    if z[k] <= 520.
      q[:θ_liq, k, gm] = 298.7
    end
    if z[k] > 520.0 && z[k] <= 1480.0
      q[:θ_liq, k, gm] = 298.7 + (z[k] - 520)  * (302.4 - 298.7)/(1480.0 - 520.0)
    end
    if z[k] > 1480.0 && z[k] <= 2000
      q[:θ_liq, k, gm] = 302.4 + (z[k] - 1480.0) * (308.2 - 302.4)/(2000.0 - 1480.0)
    end
    if z[k] > 2000.0
      q[:θ_liq, k, gm] = 308.2 + (z[k] - 2000.0) * (311.85 - 308.2)/(3000.0 - 2000.0)
    end
  end

  @inbounds for k in over_elems_real(grid)
    ts = ActiveThermoState(q, tmp, k, gm)

    tmp[:T, k, gm] = air_temperature(ts)

    # Set u profile
    if z[k] <= 700.0
      q[:u, k, gm] = -8.75
    end
    if z[k] > 700.0
      q[:u, k, gm] = -8.75 + (z[k] - 700.0) * (-4.61 - -8.75)/(3000.0 - 700.0)
    end
  end # end over_elems

  # Extrapolate to ghost points
  extrap_0th_order!(q, (:θ_liq, :q_tot, :u), grid, gm)
  extrap_0th_order!(tmp, :T, grid, gm)
  # Use grid-mean for sub-domain values:

  initialize_updrafts!(q, tmp, grid, params, dir_tree, case)
  distribute!(q, grid, (:q_tot, :θ_liq))
  distribute!(tmp, grid, (:q_liq, :T))
  diagnose_environment!(q, grid, :a, (:q_tot, :θ_liq, :w))

  # if haspkg.plots()
  #   plot_state(q, grid, dir_tree[:initial_conditions], :q_tot; i=gm)
  #   plot_state(q, grid, dir_tree[:initial_conditions], :θ_liq; i=gm)
  #   plot_state(tmp, grid, dir_tree[:initial_conditions], :T; i=gm)
  # end

end

