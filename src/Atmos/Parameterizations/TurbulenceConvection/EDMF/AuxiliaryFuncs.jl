#### AuxiliaryFuncs

export ActiveThermoState,
       ActiveThermoStateCut,
       ActiveThermoStateDual

export export_unsteady
export update_dt!
export heaviside

"""
    heaviside(x_1, x_2)

Heaviside function
 - `x_1`
 - `x_2` value specified at edge case `x_1 = 0`
"""
heaviside(x_1, x_2) = x_1==0 ? x_2 : typeof(x_1)(x_1 > 0)

"""
    ActiveThermoState(q, tmp, k, i)

Returns a `ThermodynamicState` using domain-averaged
quantities at element `k`.
"""
@inline function ActiveThermoState(q, tmp, k, i)
  return LiquidIcePotTempSHumEquil(q[:θ_liq, k, i],
                                   q[:q_tot, k, i],
                                   tmp[:ρ_0, k],
                                   tmp[:p_0, k])
end
@inline function ActiveThermoStateCut(q, tmp, k, i)
  return LiquidIcePotTempSHumEquil.(q[:θ_liq, Cut(k), i],
                                    q[:q_tot, Cut(k), i],
                                    tmp[:ρ_0, Cut(k)],
                                    tmp[:p_0, Cut(k)])
end
@inline function ActiveThermoStateDual(q, tmp, k, i)
  return LiquidIcePotTempSHumEquil.(q[:θ_liq, Dual(k), i],
                                    q[:q_tot, Dual(k), i],
                                    tmp[:ρ_0, Dual(k)],
                                    tmp[:p_0, Dual(k)])
end

function update_dt!(grid, params, q, t)
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  @unpack params Δt Δt_min
  u_max = max([q[:w, k, i] for i in ud for k in over_elems(grid)]...)
  Δt = [min(Δt_min, 0.5 * grid.Δz/max(u_max,1e-10))]
  Δti = [1/Δt[1]]
  params[:Δt] = Δt
  params[:Δti] = Δti

  # println("----------------------------")
  t[1]+=Δt[1]
  percent_done = t[1]/params[:t_end]*100.0
  # @show t[1], Δt[1], percent_done, params[:t_end]
end


"""
    export_unsteady(t, i_Δt, i_export, params, q, tmp, grid)

Exports unsteady fields
"""
function export_unsteady(t, i_Δt, i_export, params, q, tmp, grid, dir_tree)
  i_export[1]+=1
  if mod(i_export[1], 2000)==0
    directory = dir_tree[:solution_raw]*string(round(t[1], digits=5))
    println("============================")
    println("======== EXPORTING =========")
    println("============================")
    println("directory = ", directory)
    println("============================")
    println("============================")
    println("============================")
    export_plots(q, tmp, grid, directory, true, params, i_Δt)
  end
  return nothing
end

