"""
    StateVecs

Provides a state vector that is includes information
of the domain decomposition into a grid mean,
environment and updrafts.
"""
module StateVecs

using ..FiniteDifferenceGrids
export StateVec, over_sub_domains, Cut, Dual
export var_names, var_string, var_suffix
export assign!, assign_real!, assign_ghost!, extrap!, extrap_0th_order!
export compare

include("DomainDecomp.jl")
include("DomainSubSet.jl")
include("DomainIdx.jl")
include("VarMapper.jl")

struct FieldsPerElement
  vals::Vector
end

"""
    StateVec{VM}

A state vector containing the number of subdomains,
`n_subdomains`, a `NamedTuple` variable mapper, a tuple
of the variable names, and a vector of vectors, containing
the values for all of the variables.
"""
struct StateVec{VM}
  """
  A `DomainIdx`, containing the complete set of sub-domain indexes
  """
  idx::DomainIdx
  """
  A `NamedTuple` variable mapper, containing indexes per variable
  """
  var_mapper::VM
  """
  A `Dict` index per sub-domain per variable
  """
  idx_ss_per_var::Dict
  """
  A mapper, containing mapping from global sub-domain indexes to sub-domain indexes per variable
  """
  a_map::Dict
  """
  A `NamedTuple` sub-domain mapper, containing indexes per variable which are sub-domain specific
  """
  sd_unmapped::Dict
  """
  A `DomainDecomp`, which specifies the complete set of `DomainDecomp` across all variables
  """
  dd::DomainDecomp
  """
  A `DomainDecomp` per variable
  """
  dss_per_var::NamedTuple
  """
  A vector of vectors containing all values for all variables.
  """
  fields::Vector{FieldsPerElement}
end

"""
    StateVec(vars::Tuple{Vararg{Tuple{Symbol, DD}}}, grid::Grid{T}) where {DD, T}

Return a state vector, given a tuple of tuples of variable
names and the number of their subdomains.
"""
function StateVec(vars, grid::Grid{T}, dd::DomainDecomp) where {T}
  n_subdomains = sum(dd)
  n_vars = sum([sum(dd, dss) for (v, dss) in vars])
  var_mapper, dss_per_var, var_names = get_var_mapper(vars, dd)
  idx = DomainIdx(dd)
  sd_unmapped = get_sd_unmapped(vars, idx, dd)

  idx_ss_per_var = Dict(ϕ => DomainIdx(dd, dss_per_var[ϕ]) for ϕ in var_names)
  a_map = Dict(ϕ => get_sv_a_map(idx, idx_ss_per_var[ϕ]) for ϕ in var_names)

  all_vars = Any[eltype(grid.Δz)(0) for v in 1:n_vars]
  fields = [FieldsPerElement(deepcopy(all_vars)) for k in over_elems(grid)]
  return StateVec{typeof(var_mapper)}(idx,
                                      var_mapper,
                                      idx_ss_per_var,
                                      a_map,
                                      sd_unmapped,
                                      dd,
                                      dss_per_var,
                                      fields)
end

"""
    var_names(sv::StateVec)

Get tuple of variable names.
"""
var_names(sv::StateVec{VM}) where {VM} = fieldnames(VM)

function Base.show(io::IO, sv::StateVec)
  println(io, "\n----------------------- State Vector")
  println(io, "DomainIdx")
  println(io, sv.idx)
  println(io, "var_mapper")
  println(io, sv.var_mapper)
  vn = var_names(sv)
  headers = [length(over_sub_domains(sv, ϕ))==1 ? string(ϕ) : string(ϕ)*"_"*string(i)
             for ϕ in vn for i in over_sub_domains(sv, ϕ)]
  n_vars = length(headers)
  n_elem = length(sv.fields)
  data = reshape([sv[ϕ, k, i] for ϕ in vn for
    i in over_sub_domains(sv, ϕ) for k in 1:n_elem], n_elem, n_vars)
  Base.print_matrix(io, [reshape(Text.(headers), 1, n_vars); data])
  println(io, "\n-----------------------")
end

DomainIdx(sv::StateVec) = sv.idx

"""
    over_sub_domains(sv::StateVec, ϕ::Symbol)

Get list of indexes over all subdomains for variable `ϕ`.
"""
function over_sub_domains(sv::StateVec, ϕ::Symbol)
  return [x for x in sv.sd_unmapped[ϕ] if !(x==0)]
end

function Base.getindex(sv::StateVec, ϕ::Symbol, k, i = 0)
  i==0 && (i = gridmean(sv.idx))
  i_sv = get_i_state_vec(sv.var_mapper, sv.a_map[ϕ], ϕ, i)
  @inbounds sv.fields[k].vals[i_sv]
end
function Base.setindex!(sv::StateVec, val, ϕ::Symbol, k, i = 0)
  i==0 && (i = gridmean(sv.idx))
  @inbounds i_sv = get_i_state_vec(sv.var_mapper, sv.a_map[ϕ], ϕ, i)
  @inbounds sv.fields[k].vals[i_sv] = val
end

abstract type AbstractCut{I} end

"""
    Cut{I} <: AbstractCut{I}

A Cut struct used to slice the state
vector along the grid-element dimension.
This is used as an API to pass Cuts
into local derivative/interpolation routines.
"""
struct Cut{I} <: AbstractCut{I}
  e::I
end
function Base.getindex(sv::StateVec, ϕ::Symbol, cut::Cut, i=0)
  @inbounds [sv[ϕ, k, i] for k in cut.e-1:cut.e+1]
end

"""
    Dual{I} <: AbstractCut{I}

A Dual struct used to slice the state
vector along the grid-element dimension.
In addition, this interpolates fields
[(f[k-1]+f[k])/2, (f[k]+f[k+1])/2]
"""
struct Dual{I} <: AbstractCut{I}
  e::I
end
function Base.getindex(sv::StateVec, ϕ::Symbol, dual::Dual, i=0)
  @inbounds [(sv[ϕ, k, i]+sv[ϕ, k+1, i])/2 for k in dual.e-1:dual.e]
end

Base.isnan(sv::StateVec) = any([any(isnan.(fpe.vals)) for fpe in sv.fields])
Base.isinf(sv::StateVec) = any([any(isinf.(fpe.vals)) for fpe in sv.fields])


"""
    var_suffix(sv::StateVec, ϕ::Symbol, i=0)

A suffix string for variable `ϕ` indicating its sub-domain.
"""
function var_suffix(sv::StateVec, ϕ::Symbol, i=0)
  i==0 && (i = gridmean(DomainIdx(sv)))
  var_suffix(sv.var_mapper, sv.idx, sv.idx_ss_per_var[ϕ], ϕ, i)
end

"""
    var_string(sv::StateVec, ϕ::Symbol, i=0)

A string of the variable `ϕ` with the appropriate suffix indicating its sub-domain.
"""
function var_string(sv::StateVec, ϕ::Symbol, i=0)
  i==0 && (i = gridmean(DomainIdx(sv)))
  var_string(sv.var_mapper, sv.idx, sv.idx_ss_per_var[ϕ], ϕ, i)
end

"""
    assign!(sv::StateVec, var_names, grid::Grid, val, i=1)

Assign value `val` to variable `ϕ` for all ghost points.
"""
function assign!(sv::StateVec, var_names, grid::Grid{FT}, val::FT) where FT
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  !(var_names isa Tuple) && (var_names = (var_names,))
  @inbounds for k in over_elems(grid), ϕ in var_names, i in over_sub_domains(sv, ϕ)
    sv[ϕ, k, i] = val
  end
end

"""
    assign!(sv::StateVec, grid::Grid, val)

Assign value `val` to all variables in state vector.
"""
function assign!(sv::StateVec, grid::Grid{FT}, val::FT) where FT
  gm, en, ud, sd, al = allcombinations(DomainIdx(sv))
  @inbounds for k in over_elems(grid), ϕ in var_names(sv), i in over_sub_domains(sv, ϕ)
    sv[ϕ, k, i] = val
  end
end

"""
    assign_real!(sv::StateVec, ϕ::Symbol, grid::Grid, vec_real, i=0)

Assign the real elements to the given vector `vec_real`.
"""
function assign_real!(sv::StateVec, ϕ::Symbol, grid::Grid, vec_real, i=0)
  i==0 && (i = gridmean(sv.idx))
  k_real = 1
  @assert length(vec_real)==length(over_elems_real(grid))
  @inbounds for k in over_elems_real(grid)
    sv[ϕ, k, i] = vec_real[k_real]
    k_real+=1
  end
end

"""
    assign_ghost!(sv::StateVec, ϕ::Symbol, grid::Grid, val, i=0)

Assign value `val` to variable `ϕ` for all ghost points.
"""
function assign_ghost!(sv::StateVec, ϕ::Symbol, grid::Grid, val, i=0)
  i==0 && (i = gridmean(sv.idx))
  @inbounds for k in over_elems_ghost(grid)
    sv[ϕ, k, i] = val
  end
end

"""
    assign_ghost!(sv::StateVec, ϕ::Symbol, grid::Grid, val, b::ZBoundary, i=1)

Assign value `val` to variable `ϕ` for all ghost points.
"""
function assign_ghost!(sv::StateVec, ϕ::Symbol, grid::Grid, val, b::ZBoundary, i=0)
  i==0 && (i = gridmean(sv.idx))
  @inbounds for k in over_elems_ghost(grid, b)
    sv[ϕ, k, i] = val
  end
end

"""
    compare(sv::StateVec, sv_expected::StateVec, grid::Grid, tol)

A dictionary, with keys in `var_names(vs)`, containing a `Bool` indicating
that `sv` ≈ `sv_expected` for all of their sub-domains for all elements.
"""
function compare(sv::StateVec, sv_expected::StateVec, grid::Grid{FT}, tol) where FT
  D = Dict(ϕ => [true for i in over_sub_domains(sv, ϕ)] for ϕ in var_names(sv))
  @inbounds for k in over_elems(grid)
    @inbounds for ϕ in var_names(sv)
      @inbounds for i in over_sub_domains(sv, ϕ)
        if abs(sv[ϕ, k, i] - sv_expected[ϕ, k, i]) > tol
          i_var = get_i_var(sv.a_map[ϕ], i)
          D[ϕ][i_var] = false
        end
      end
    end
  end
  return D
end

"""
    extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, i=1)

Extrapolate variable `ϕ` to the first ghost point.
"""
function extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, i=0)
  i==0 && (i = gridmean(sv.idx))
  @inbounds for b in (Zmin(), Zmax())
    kg, ki, kii = boundary_points(grid, b)
    sv[ϕ, kg, i] = 2*sv[ϕ, ki, i] - sv[ϕ, kii, i]
  end
end

"""
    extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, ::ZBoundary, i=1)

Extrapolate variable `ϕ` to the first ghost point.
"""
function extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, b::ZBoundary, i=0)
  i==0 && (i = gridmean(sv.idx))
  kg, ki, kii = boundary_points(grid, b)
  @inbounds sv[ϕ, kg, i] = 2*sv[ϕ, ki, i] - sv[ϕ, kii, i]
end

"""
    extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, dual_val, b::ZBoundary, i=1)

Extrapolate variable `ϕ` to the first ghost point.
"""
function extrap!(sv::StateVec, ϕ::Symbol, grid::Grid, dual_val, b::ZBoundary, i=0)
  i==0 && (i = gridmean(sv.idx))
  ki = first_interior(grid, b)
  kg = first_ghost(grid, b)
  @inbounds sv[ϕ, kg, i] = 2*dual_val - sv[ϕ, ki, i]
end

"""
    extrap_0th_order!(sv::StateVec, ϕ::Symbol, grid::Grid, i=1)

Extrapolate variable `ϕ` to the first ghost point using zeroth order approximation.
"""
function extrap_0th_order!(sv::StateVec, ϕ::Symbol, grid::Grid, i=0)
  i==0 && (i = gridmean(sv.idx))
  @inbounds for b in (Zmin(), Zmax())
    kg, ki, kii = boundary_points(grid, b)
    sv[ϕ, kg, i] = sv[ϕ, ki, i]
  end
end

include("BoundaryConditions.jl")
include("ExportFuncs.jl")
include("PlotFuncs.jl")
include("DomainDecompFuncs.jl")

end
