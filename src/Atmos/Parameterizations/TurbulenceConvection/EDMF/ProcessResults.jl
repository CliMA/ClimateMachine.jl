#### ProcessResults

const plot_single_fields = false
const export_raw_data = false

export init_netcdf,
       export_profiles!,
       export_referece!,
       export_time_series!

function export_initial_conditions(q, tmp, grid, directory, include_ghost)
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  @static if haspkg("Plots")
    N = grid.n_elem
    @inbounds for i in al
      plot_state(q  , grid, directory, :q_tot     , i=i, include_ghost=include_ghost, i_Δt=N)
      plot_state(q  , grid, directory, :θ_liq     , i=i, include_ghost=include_ghost, i_Δt=N)
      plot_state(tmp, grid, directory, :T         , i=i, include_ghost=include_ghost, i_Δt=N)
    end
  end
end

function export_data(q, tmp, grid, dir_tree)
  export_state(q, grid, dir_tree.output, "q.csv")
  export_state(tmp, grid, dir_tree.output, "tmp.csv")
end


struct NetIOHelper
  t_domain
  z_domain
  profile
  reference
  time_series
end

using NCDatasets
using NetCDF

const refernce_var_names = (:ρ_0, :α_0, :p_0)
const time_series_var_names = (:time, )

function init_netcdf(q, tmp, grid, dir_tree, include_ghost, params, i_Δt)
  @unpack params t_end Δt
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  t_domain = 1:ceil(t_end/Δt)
  z_domain = include_ghost ? over_elems(grid) : over_elems_real(grid)
  mkpath(dir_tree.output)
  file_name = joinpath(dir_tree.output, "test.nc")
  try
    rm(file_name)
  catch
  end
  println("prepping file ", file_name)
  ds = Dataset(file_name, "c")
  T = eltype(grid.zc)
  defDim(ds, "t", length(t_domain))
  defDim(ds, "z", length(z_domain))
  defGroup(ds, "Profiles")
  defGroup(ds, "Reference")
  defGroup(ds, "TimeSeries")
  profile_vars     = Dict([Symbol(ϕ, :_, i) => defVar(ds, string(Symbol(ϕ, :_, i)), T, ("t", "z")) for ϕ in var_names(q) for i in sd]...)
  reference_vars   = Dict([ϕ => defVar(ds, string(ϕ), T, ("z",)) for ϕ in refernce_var_names]...)
  time_series_vars = Dict([ϕ => defVar(ds, string(ϕ), T, ("t",)) for ϕ in time_series_var_names]...)
  ds_io = NetIOHelper(t_domain, z_domain, profile_vars, reference_vars, time_series_vars)
  return ds_io, ds
end

""" 2D(t, z) """
function export_profiles!(ds_io, ds, q, tmp, grid, i_Δt)
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  @inbounds for i in sd
    @inbounds for ϕ in var_names(q)
      ϕ_i = Symbol(ϕ, :_, i)
      ds_io.profile[ϕ_i][i_Δt, :] .= [q[ϕ, k, i] for k in ds_io.z_domain]
    end
  end
end

""" 1D(z) """
function export_referece!(ds_io, ds, q, tmp, grid, i_Δt)
  @inbounds for ϕ in refernce_var_names
    ds_io.reference[ϕ][:] .= [tmp[ϕ, k] for k in ds_io.z_domain]
  end
end

""" 1D(t) """
function export_time_series!(ds_io, ds, q, tmp, grid, i_Δt, t)
  @inbounds for ϕ in time_series_var_names
    ds_io.time_series[ϕ][i_Δt] = t
  end
end

function export_plots(q, tmp, grid, directory, include_ghost, params, i_Δt)
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  @static if haspkg("Plots")

    dsf = joinpath(directory,"SingleFields")
    dcf = joinpath(directory,"CombinedFields")
    ds = joinpath(directory,"Sources")
    mkpath(dsf)
    mkpath(dcf)
    mkpath(ds)
    qv = (:a, :w, :q_tot, :θ_liq, :tke)
    tv = (:HVSD_w, :HVSD_a, :buoy, :T, :K_m, :K_h, :l_mix)
    if plot_single_fields
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
