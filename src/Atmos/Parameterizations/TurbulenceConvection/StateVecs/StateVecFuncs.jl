"""
    StateVecFuncs

  Provides a set of helper functions that operate on StateVec.
"""
module StateVecFuncs

using Requires
@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
  using .Plots
  export plot_state
end

using DelimitedFiles, WriteVTK, ..Grids, ..StateVecs
export surface_val, first_elem_above_surface_val
export domain_average!, distribute!, total_covariance!
export extrap!, assign_ghost!, integrate_ode!
export export_state, UseVTK, UseDat

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

Formulaically, a domain-averaged variable ``⟨ϕ⟩`` is computed from

``⟨ϕ⟩ = Σ_i a_i \\overline{ϕ}_i``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.

Note that `domain_average!` is the inverse function of `distribute!`.
"""
function domain_average!(dst::StateVec, src::StateVec, weight::StateVec,
                         dst_idxs, src_idxs, weight_idx, grid::Grid)
  @inbounds for k in over_elems(grid)
    @inbounds for (dst_idx, src_idx) in zip(dst_idxs, src_idxs)
      dst[dst_idx, k] = 0
      @inbounds for i in over_sub_domains(src)
        dst[dst_idx, k] += src[src_idx, k, i]*weight[weight_idx, k, i]
      end
    end
  end
end

"""
    distribute!(dst::StateVec, src::StateVec, dst_idxs, src_idxs, grid::Grid)

Distributes values in the state vector `src`, to state vectors `dst` given
the grid `grid` and index iterators of source and destination names `src_idxs`
and `dst_idxs` respectively.

Formulaically, a domain-decomposed variable ``\\overline{ϕ}_i`` is computed from

``\\overline{ϕ}_i = ⟨ϕ⟩``

Where variable ``⟨ϕ⟩`` is the domain-averaged variable, computed
across multiple sub-domains.

Note that `distribute!` is the inverse function of `domain_average!`.
"""
function distribute!(dst::StateVec, src::StateVec, dst_idxs, src_idxs, grid::Grid)
  @inbounds for k in over_elems(grid), i in over_sub_domains(dst)
    @inbounds for (dst_idx, src_idx) in zip(dst_idxs, src_idxs)
      dst[dst_idx, k, i] = src[src_idx, k]
    end
  end
end

"""
    total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                      dst_idxs, src_idxs, cv_idxs, weight_idx,
                      grid::Grid, decompose_ϕ_ψ::Function)

Computes the total covariance in state vector `dst`, given
 - `src` source state vector
 - `cv` state vector containing co-variances
 - `weights` state vector containing weights
 - `dst_idxs` indexes for destination state vector
 - `cv_idxs` indexes for state vector containing co-variances
 - `weight_idx` indexes for state vector containing weights
 - `grid` the grid
 - `decompose_ϕ_ψ` a function that receives the covariance index and
                   returns the indexes for each variable. For example:
                   `:ϕ_idx, :ψ_idx = decompose_ϕ_ψ(:cv_ϕ_ψ)`

Formulaically, a total covariance between variables ``ϕ`` and ``ψ`` is computed from

``⟨ϕ^*ψ^*⟩ = Σ_i a_i \\overline{ϕ_i'ψ_i'} + Σ_i Σ_j a_i a_j \\overline{ϕ}_i (\\overline{ψ}_i - \\overline{ψ}_j)``

Where variable ``\\overline{ϕ}_i`` represents ``ϕ`` decomposed across multiple
sub-domains, which are weighted by area fractions ``a_i``.
"""
function total_covariance!(dst::StateVec, src::StateVec, cv::StateVec, weights::StateVec,
                           dst_idxs, cv_idxs, weight_idx, grid::Grid, decompose_ϕ_ψ::Function)
  @inbounds for k in over_elems(grid), (dst_idx, cv_idx) in zip(dst_idxs, cv_idxs)
    _ϕ, _ψ = decompose_ϕ_ψ(cv_idx)
    dst[dst_idx, k] = 0
    @inbounds for i in over_sub_domains(weights)
      dst[dst_idx, k] += weights[weight_idx, k, i]*cv[cv_idx, k, i]
      @inbounds for j in over_sub_domains(src)
        dst[dst_idx, k] += weights[weight_idx, k, i]*
                           weights[weight_idx, k, j]*
                           src[_ϕ, k, i]*(src[_ψ, k, i] - src[_ψ, k, j])
      end
    end
  end
end

"""
    extrap!(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)

Extrapolate variable `name` to the first ghost point.
"""
function extrap!(sv::StateVec, name::Symbol, grid::Grid, i_sd=1)
  k = 1+grid.n_ghost
  sv[name, k-1, i_sd] = 2*sv[name, k, i_sd] - sv[name, k+1, i_sd]
  k = grid.n_elem - grid.n_ghost
  sv[name, k+1, i_sd] = 2*sv[name, k, i_sd] - sv[name, k-1, i_sd]
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

Integrates the Ordinary Differential Equation

  ``\\frac{dy}{dz} = f(z)``

using Newton method, given the arguments

 - `sv` a state vector
 - `name` the name of the variable integrated
 - `grid` the grid that the variable lives on
 - `func` the function ``f(z)`` to be integrated, which accepts arguments `func(z, args)`
 - `y_0` the boundary condition ``y|_{z=0} = y_0``
 - `args` a `NamedTuple` of arguments passed to `func`
and optionally,
 - `i_sd` the i-th sub-domain of the variable (default is 1)
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
  @inbounds for k in over_elems_real(grid)
    f = func(get_z(grid, k), args)
    sv[name, k, i_sd] = sv[name, k-1, i_sd] + grid.dz * f
  end
  @inbounds for k in over_elems_real(grid)
    sv[name, k, i_sd] += y_0
  end
  assign_ghost!(sv, name, 0.0, grid, i_sd)
end

abstract type ExportType end
"""
    UseVTK

A singleton used to indicate to use a `.vtk`
file extension when exporting a `StateVec`.
"""
struct UseVTK <: ExportType end

"""
    UseDat

A singleton used to indicate to use a `.dat` (DAT)
file extension when exporting a `StateVec`. A DAT
file is a generic data file which may contain data
in binary or text format.
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
    export_state(sv::StateVec, dir, filename, ::ExportType = UseDat())

Export StateVec to a human-readable file `filename` in directory `dir`.
The `z`-axis is _not_ included in the export.
"""
function export_state(sv::StateVec, dir, filename, ::ExportType = UseDat())
  open(string(dir, filename*".dat"),"w") do file
    print(file, sv)
  end
end

"""
    export_state(sv::StateVec, grid::Grid, dir, filename, ::UseVTK)

Export state vector `sv` to a compressed file `filename` in directory `dir`,
including the `z`-axis, given by the grid `grid`.
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
  @inbounds for k in domain
    cells[k] = MeshCell(VTKCellTypes.VTK_LINE, [k, k+1])
  end
  vtkfile = vtk_grid("$(filename)", z, cells; compress=false)
  @inbounds for (name, v) in fields
    vtk_point_data(vtkfile, v, name)
  end
  outfiles = vtk_save(vtkfile)
end

@init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin

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
                    include_ghost = true,
                    xlims::Union{Nothing, Tuple{R, R}} = nothing,
                    ylims::Union{Nothing, Tuple{R, R}} = nothing
                    ) where R
  if name_idx == nothing
    domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
    @inbounds for name_idx in sv.var_names
      x = [grid.z[k] for k in domain_range]
      y = [sv[name_idx, k, i_sd] for k in domain_range]
      plot(y, x)
    end
    plot!(title = "state vector vs z", xlabel = "state vector", ylabel = "z")
    if xlims != nothing; plot!(xlims = xlims); end
    if ylims != nothing; plot!(ylims = xlims); end
  else
    x_name = filename
    domain_range = include_ghost ? over_elems(grid) : over_elems_real(grid)
    x = [grid.z[k] for k in domain_range]
    y = [sv[name_idx, k, i_sd] for k in domain_range]
    plot(y, x)
    plot!(title = x_name * " vs z", xlabel = x_name, ylabel = "z")
    if xlims != nothing; plot!(xlims = xlims); end
    if ylims != nothing; plot!(ylims = xlims); end
  end
  savefig(joinpath(directory, filename))
end

end

end