#### ProcessResults

# TODO: Migrate to using NetCDF IO
if haspkg.plots()
using Plots
end
using NCDatasets

function export_data(q, tmp, grid, dir_tree, params)
  if params[:export_data]
    export_state(q, grid, dir_tree.output, "q.csv")
    export_state(tmp, grid, dir_tree.output, "tmp.csv")
  end
end

function export_plots(q, tmp, grid, directory, include_ghost, params, i_Δt)
  if haspkg.plots()
    gm, en, ud, sd, al = allcombinations(q)
    dsf = joinpath(directory,"SingleFields")
    dcf = joinpath(directory,"CombinedFields")
    ds = joinpath(directory,"Sources")
    mkpath(dsf)
    mkpath(dcf)
    mkpath(ds)
    qv = (:a, :w, :q_tot, :θ_liq, :tke)
    tv = (:buoy, :T, :K_m, :K_h, :l_mix, :q_liq, :ε_model, :δ_model)
    if params[:plot_single_fields]
      @inbounds for v in qv, i in over_sub_domains(q, v)
        plot_state(q, grid, dsf, v, i=i, i_Δt=i_Δt)
      end
      @inbounds for v in tv, i in over_sub_domains(tmp, v)
        plot_state(tmp, grid, dsf, v, i=i, i_Δt=i_Δt)
      end
    end
    @inbounds for v in qv
      plot_state(q, grid, dcf, v, i_Δt=i_Δt)
    end
    @inbounds for v in tv
      plot_state(tmp, grid, dcf, v, i_Δt=i_Δt)
    end
  end
end

function export_initial_conditions(q, tmp, grid, directory, include_ghost)
  if haspkg.plots()
    gm, en, ud, sd, al = allcombinations(q)
    N = grid.n_elem
    @inbounds for i in al
      plot_state(q  , grid, directory, :q_tot     , i=i, include_ghost=include_ghost, i_Δt=N)
      plot_state(q  , grid, directory, :θ_liq     , i=i, include_ghost=include_ghost, i_Δt=N)
      plot_state(tmp, grid, directory, :T         , i=i, include_ghost=include_ghost, i_Δt=N)
    end
  end
end

