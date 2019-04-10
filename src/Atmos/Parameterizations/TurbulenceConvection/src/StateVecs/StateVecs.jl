"""
    StateVecs

Provides a state vector given a `NamedTuple` containing
tuples of the variable name and the number of its components.
"""
module StateVecs

using ..Grids
export StateVec, over_sub_domains, Slice

struct FieldsPerElement{T}
  vals::Vector{T}
end

function Base.show(io::IO, fpe::FieldsPerElement)
  print(io, "fpe = ",fpe.vals)
end

"""
    StateVec{T, I, NT}

A state vector containing the number of subdomains,
`n_subdomains`, a `NamedTuple` variable mapper, a tuple
of the variable names, and a vector of vectors, containing
the values for all of the variables.
"""
struct StateVec{T, I, NT}
  """
  Number of subdomains.
  """
  n_subdomains::I

  """
  A `NamedTuple` that maps the variable name to its unique index.
  """
  var_mapper::NT

  """
  A tuple of all variable names.
  """
  var_names::Tuple{Vararg{Symbol}}

  """
  A vector of vectors containing all values for all variables.
  """
  fields::Vector{FieldsPerElement{T}}
end

"""
    get_var_mapper(vars)

Get a `NamedTuple` that maps a variable name
and subdomain ID to a unique index. Example:
```
vars = ((:ρ_0, 1), (:w, 3), (:a, 3), (:α_0, 1))
var_mapper = get_var_mapper(vars)
((:ρ_0, :w, :a, :α_0), (ρ_0 = [1], w = [2, 3, 4], a = [5, 6, 7], α_0 = [8]))
```
"""
function get_var_mapper(vars)
  var_names = tuple([v for (v, nsd) in vars]...)
  end_index = cumsum([nsd for (v, nsd) in vars])
  start_index = [1,[1+x for x in end_index][1:end-1]...]
  vals = [collect(a:b) for (a,b) in zip(start_index, end_index)]
  var_mapper = NamedTuple{var_names}(vals)
  return var_names, var_mapper
end

"""
    StateVec(vars::Tuple{Vararg{Tuple{Symbol, I}}}, grid::Grid{T}) where {I, T}

Return a state vector, given a tuple of tuples of variable
names and the number of their subdomains.
"""
function StateVec(vars::Tuple{Vararg{Tuple{Symbol, I}}}, grid::Grid{T}) where {I, T}
  n_subdomains = max([nsd for (v, nsd) in vars]...)
  n_vars = sum([nsd for (v, nsd) in vars])
  var_names, var_mapper = get_var_mapper(vars)
  all_vars = [eltype(grid.dz)(0) for v in 1:n_vars]
  fields = [FieldsPerElement(deepcopy(all_vars)) for k in over_elems(grid)]
  return StateVec(n_subdomains, var_mapper, var_names, fields)
end

function Base.show(io::IO, sv::StateVec)
  println(io, "----------------------- State Vector")
  println(io, "n_subdomains    = ", sv.n_subdomains)
  println(io, "var_mapper      = ", sv.var_mapper)
  for fpe in sv.fields
    println(io, fpe)
  end
  println(io, "-----------------------")
end

"""
    over_sub_domains(state_vec::StateVec)

Get list of indexes from 1 to the maximum subdomain size.
"""
over_sub_domains(state_vec::StateVec) = 1:state_vec.n_subdomains

"""
    over_sub_domains(state_vec::StateVec, j::Int)

Get list of indexes from 1 to the maximum subdomain size, except the given index.
"""
over_sub_domains(state_vec::StateVec, j::Int) = [i for i in 1:state_vec.n_subdomains if !(i==j)]

"""
    over_sub_domains(state_vec::StateVec, name::Symbol)

Get list of indexes over all subdomains for variable `name`.
"""
over_sub_domains(state_vec::StateVec, name::Symbol) = 1:length(state_vec.var_mapper[name])


Base.getindex(sv::StateVec, name::Symbol, k, i_sd=1) = sv.fields[k].vals[sv.var_mapper[name][i_sd]]
function Base.setindex!(sv::StateVec, val, name::Symbol, k, i_sd = 1)
  sv.fields[k].vals[sv.var_mapper[name][i_sd]] = val
end

abstract type AbstractSlice{I} end

"""
    Slice{I} <: AbstractSlice{I}

A slice struct used to slice the state
vector along the grid-element dimension.
This is used to as an API to pass slices
into local derivative/interpolation routines.
"""
struct Slice{I} <: AbstractSlice{I}
  e::I
end
Base.getindex(sv::StateVec, name::Symbol, slice::Slice, i_sd=1) = [sv[name, k, i_sd] for k in slice.e-1:slice.e+1]

Base.isnan(sv::StateVec) = any([any(isnan.(fpe.vals)) for fpe in sv.fields])
Base.isinf(sv::StateVec) = any([any(isinf.(fpe.vals)) for fpe in sv.fields])

end