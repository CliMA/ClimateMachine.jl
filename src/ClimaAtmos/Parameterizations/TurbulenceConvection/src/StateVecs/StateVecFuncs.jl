"""
    StateVecFuncs

  Provides a set of helper functions that operate on StateVec.
"""
module StateVecFuncs

using Pkg
@static if haskey(Pkg.installed(), "Plots")
  using Plots
  export plot_state
end

using DelimitedFiles, WriteVTK, ..Grids, ..StateVecs
export surface_val, first_elem_above_surface_val
export domain_average!, distribute!, total_covariance!
export extrap!, assign_ghost!, integrate_ode!
export export_state, plot_state, UseVTK, UseDat

"""
    surface_val(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)

Get the value of variable `name` on the surface (`z_min`) by
interpolating between the first ghost point and the first element
above the surface
"""
function surface_val(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)
  return (sv[name, grid.n_ghost, i_sd]+sv[name, 1+grid.n_ghost, i_sd])/2
end

"""
    first_elem_above_surface_val(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)

Get the value of variable `name` on the first element above the
surface (`z_min+Δz/2`).
"""
function first_elem_above_surface_val(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)
  return sv[name, 1+grid.n_ghost, i_sd]
end

"""
    domain_average!(dst::StateVec, src::StateVec, weight::StateVec
                    dst_idxs, src_idxs, weight_idx, grid::Grid)

Compute the domain average in state vector `dst`, given state vectors `src`
and `weight`, the grid `grid` and index iterators of source, destination and
weight names `src_idxs`, `dst_idxs`, and `weight_idx` respectively.
"""
function domain_average!(dst::StateVec, src::StateVec, weight::StateVec,
                         dst_idxs, src_idxs, weight_idx, grid::Grid)
  for k in over_elems(grid)
    for (dst_idx, src_idx) in zip(dst_idxs, src_idxs)
      dst[dst_idx, k] = 0
      for i in over_sub_domains(src)
        dst[dst_idx, k] += src[src_idx, k, i]*weight[weight_idx, k, i]
      end
    end
  end
end

"""
    distribute!(dst::StateVec, src::StateVec, weight::StateVec
                dst_idxs, src_idxs, weight_idx, grid::Grid)

Distributes values in the state vector `src`, to state vectors `dst` given
`weight` state vector, the grid `grid` and index iterators of source,
destination and weight names `src_idxs`, `dst_idxs`, and `weight_idx`
respectively.
"""
function distribute!(dst::StateVec, src::StateVec, weight::StateVec,
                     dst_idxs, src_idxs, weight_idx, grid::Grid)
  for k in over_elems(grid), i in over_sub_domains(dst)
    for (dst_idx, src_idx) in zip(dst_idxs, src_idxs)
      dst[dst_idx, k, i] = src[src_idx, k]/weight[weight_idx, k, i]
    end
  end
end

"""
    distribute!(dst::StateVec, src::StateVec, dst_idxs, src_idxs, grid::Grid)

Distributes values in the state vector `src`, to state vectors `dst` given
the grid `grid` and index iterators of source and destination names `src_idxs`
and `dst_idxs` respectively.
"""
function distribute!(dst::StateVec, src::StateVec, dst_idxs, src_idxs, grid::Grid)
  for e in over_elems(grid), i in over_sub_domains(dst)
    for (dst_idx, src_idx) in zip(dst_idxs, src_idxs)
      dst[dst_idx, e, i] = src[src_idx, e]
    end
  end
end

"""
    total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                      dst_idxs, src_idxs, cv_idxs, weight_idx, grid::Grid, decompose_ϕ_ψ::Function)

Computes the total covariance in state vector `dst`, given
 - `src` source state vector
 - `cv` state vector containing co-variances
 - `weights` state vector containing weights
 - `dst_idxs` indexes for destination state vector
 - `cv_idxs` indexes for state vector containing co-variances
 - `weight_idx` indexes for state vector containing weights
 - `grid` the grid
 - `decompose_ϕ_ψ` a function that receives accepts the covariance index
                   and returns the indexes for each variable. For example:
                   :ϕ_idx, :ψ_idx = decompose_ϕ_ψ(:covar_ϕ_ψ)
"""
function total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                           dst_idxs, cv_idxs, weight_idx, grid::Grid, decompose_ϕ_ψ::Function)
  for e in over_elems(grid), (dst_idx, cv_idx) in zip(dst_idxs, cv_idxs)
    _ϕ, _ψ = decompose_ϕ_ψ(cv_idx)
    dst[dst_idx, e] = 0
    for i in over_sub_domains(weights)
      dst[dst_idx, e] += weights[weight_idx, e, i]*cv[cv_idx, e, i]
      for j in over_sub_domains(src)
        dst[dst_idx, e] += weights[weight_idx, e, i]*
                           weights[weight_idx, e, j]*
                           src[_ϕ, e, i]*(src[_ψ, e, i] - src[_ψ, e, j])
      end
    end
  end
end

"""
    extrap!(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)

Extrapolate variable `name` to the first ghost point.
"""
function extrap!(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)
  e = 1+grid.n_ghost
  sv[name, e-1, i_sd] = 2*sv[name, e, i_sd] - sv[name, e+1, i_sd]
  e = grid.n_elem - grid.n_ghost
  sv[name, e+1, i_sd] = 2*sv[name, e, i_sd] - sv[name, e-1, i_sd]
end

"""
    assign_ghost!(sv::StateVec, name::Symbol, val, grid::Grid, i_sd=1)

Assign value `val` to variable `name` for all ghost points.
"""
function assign_ghost!(sv::StateVec, name::Symbol, val, grid::Grid, i_sd=1)
  for k in over_elems_ghost(grid)
    sv[name, k, i_sd] = val
  end
end

"""
    integrate_ode!(sv::StateVec,
                   name::Symbol,
                   grid::Grid,
                   func::Function,
                   y_0::AbstractFloat,
                   args::NamedTuple,
                   i_sd::Int = 1)

Integrates the ODE:
  `dy/dz = func(z, args)`
  `y = y_0 + int_{z=0}^{z} func(z, args) dz`
"""
function integrate_ode!(sv::StateVec,
                        name::Symbol,
                        grid::Grid,
                        func::Function,
                        y_0::AbstractFloat,
                        args::NamedTuple,
                        i_sd::Int = 1)
  k = first_elem_above_surface(grid)
  sv[name, k-1, i_sd] = 0
  for k in over_elems_real(grid)
    f = func(get_z(grid, k), args)
    sv[name, k, i_sd] = sv[name, k-1, i_sd] + grid.dz * f
  end
  for k in over_elems_real(grid)
    sv[name, k, i_sd] += y_0
  end
  assign_ghost!(sv, name, 0.0, grid, i_sd)
end

abstract type ExportType end
"""
    UseVTK
A singleton used to indicate to use a .vtk
extension when exporting a `StateVec`.
"""
struct UseVTK <: ExportType end

"""
    UseDat
A singleton used to indicate to use a .dat
extension when exporting a `StateVec`.
"""
struct UseDat <: ExportType end

"""
    export_state(sv::StateVec, grid::Grid, dir, filename, ::ExportType)

Export StateVec to a human-readable file `filename` in directory `dir`.
"""
function export_state(sv::StateVec, grid::Grid, dir, filename, ::ExportType = UseDat())
  domain = over_elems(grid)
  headers = [ length(over_sub_domains(sv, name))==1 ? string(name) : string(name)*"_"*string(i_sd)
             for name in sv.var_names for i_sd in over_sub_domains(sv, name)]
  n_vars = length(headers)
  n_elem = length(domain)
  data = reshape([sv[name, k, i_sd] for name in sv.var_names for
    i_sd in over_sub_domains(sv, name) for k in domain], n_elem, n_vars)
  z = grid.z[domain]
  data_all = hcat(z, data)
  open(string(dir, filename*"_vs_z.dat"),"w") do file
    write(file, join(headers, ", ")*"\n")
    writedlm(file, data_all)
  end
end

"""
    export_state(sv::StateVec, grid::Grid, dir, filename, ::UseVTK)

Export StateVec to a compressed file `filename` in directory `dir`.
"""
function export_state(sv::StateVec, grid::Grid, dir, filename, ::UseVTK)
  domain = over_elems(grid)
  headers = [ length(over_sub_domains(sv, name))==1 ? string(name) : string(name)*"_"*string(i_sd)
             for name in sv.var_names for i_sd in over_sub_domains(sv, name)]
  n_vars = length(headers)
  n_elem = length(domain)
  data = reshape([sv[name, k, i_sd] for name in sv.var_names for
    i_sd in over_sub_domains(sv, name) for k in domain], n_elem, n_vars)
  z = grid.z[domain]
  fields = ((headers[i], data[:,i]) for i in 1:n_vars)
  points = [1, length(domain)]
  cells = Array{MeshCell{Array{Int,1}}, 1}(undef, n_elem)
  for k in domain
    cells[k] = MeshCell(VTKCellTypes.VTK_LINE, [k, k+1])
  end
  vtkfile = vtk_grid("$(filename)", z, cells; compress=false)
  for (name, v) in fields
    vtk_point_data(vtkfile, v, name)
  end
  outfiles = vtk_save(vtkfile)
end

@static if haskey(Pkg.installed(), "Plots")

"""
    plot_state(sv::StateVec,
                    grid::Grid,
                    name_idx::Symbol = nothing,
                    directory::AbstractString,
                    filename::AbstractString,
                    i_sd = 1,
                    include_ghost = false,
                    xlims::Union{Nothing, Tuple{R, R}} = nothing,
                    ylims::Union{Nothing, Tuple{R, R}} = nothing
                    ) where R

Save the plot variable along the z-direction in `StateVec` given the
grid, `grid`, variable name `name_idx`, directory `directory`,
filename `filename`, sub-domain `i_sd`, and a `Bool`, `include_ghost`,
indicating to include include or exclude the ghost points.
"""
function plot_state(sv::StateVec,
                    grid::Grid,
                    directory::AbstractString,
                    filename::AbstractString,
                    name_idx::Symbol = nothing,
                    i_sd = 1,
                    include_ghost = false,
                    xlims::Union{Nothing, Tuple{R, R}} = nothing,
                    ylims::Union{Nothing, Tuple{R, R}} = nothing
                    ) where R
  if name_idx == nothing
    domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
    for name_idx in sv.var_names
      x = [grid.z[k] for k in domain_range]
      y = [sv[name_idx, k, i_sd] for k in domain_range]
      plot(y, x)
    end
    plot!(title = "state vector vs z", xlabel = "state vector", ylabel = "z")
    if xlims != nothing; plot!(xlims = xlims); end
    if ylims != nothing; plot!(ylims = xlims); end
    png(joinpath(directory, filename))
  else
    x_name = string(name_idx)
    domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
    x = [grid.z[k] for k in domain_range]
    y = [sv[name_idx, k, i_sd] for k in domain_range]
    plot(y, x)
    plot!(title = x_name * " vs z", xlabel = x_name, ylabel = "z")
    if xlims != nothing; plot!(xlims = xlims); end
    if ylims != nothing; plot!(ylims = xlims); end
    png(joinpath(directory, filename))
  end
end

end

end