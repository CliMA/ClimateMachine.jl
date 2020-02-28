#### RefState

using OrdinaryDiffEq

"""
    init_ref_state!(tmp::StateVec,
                    grid::Grid,
                    params,
                    dir_tree::DirTree)

Initializes the reference state variables:
  - `p_0` pressure
  - `ρ_0` density
  - `α_0` specific volume
assuming a hydrostatic balance.
FIXME: add reference
"""
function init_ref_state!(tmp::StateVec, grid::Grid, params, dir_tree::DirTree)
  @unpack params qtg Tg Pg
  q_pt_g = PhasePartition(qtg)
  θ_liq_ice_g = liquid_ice_pottemp_given_pressure(Tg, Pg, q_pt_g)
  logp = log(Pg)

  function tendencies(p, u, z)
    expp = exp(p)
    ρ = air_density(Tg, expp, q_pt_g)
    ts = LiquidIcePotTempSHumEquil_old(θ_liq_ice_g, qtg, ρ, expp)
    R_m = gas_constant_air(ts)
    T = air_temperature(ts)
    return - grav / (T * R_m)
  end

  z_span = (grid.zn_min, grid.zn_max)
  prob = ODEProblem(tendencies, logp, z_span)
  sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
  p_0 = [exp(sol(grid.zc[k])) for k in over_elems_real(grid)]
  assign_real!(tmp, :p_0, grid, p_0)
  apply_Neumann!(tmp, :p_0, grid, 0.0, Zmin())
  apply_Neumann!(tmp, :p_0, grid, 0.0, Zmax())

  @inbounds for k in over_elems(grid)
    ts = TemperatureSHumEquil(Tg, tmp[:p_0, k], qtg)
    q_pt = PhasePartition(ts)
    T = air_temperature(ts)
    tmp[:ρ_0, k] = air_density(T, tmp[:p_0, k], q_pt)
    tmp[:α_0, k] = 1/tmp[:ρ_0, k]
  end
  extrap!(tmp, :ρ_0, grid)
  extrap!(tmp, :α_0, grid)
  extrap!(tmp, :p_0, grid)

  # @if haspkg.plots()
  #   plot_state(tmp, grid, dir_tree[:initial_conditions], :p_0)
  #   plot_state(tmp, grid, dir_tree[:initial_conditions], :ρ_0)
  #   plot_state(tmp, grid, dir_tree[:initial_conditions], :α_0)
  # end
end

