"""
StateVecs

  Provides a state vector for a generalized set of fields.

# Interface
 - [`FieldsPerElement`](@ref) a set of fields that exist at each grid element
 - [`StateVec`](@ref) contains a FieldsPerElement for every mesh point, and a name mapping dictionary

"""
module StateVecs

export StateVec
export over_sub_domains

using Grids
using Test
using StaticArrays
import Base

struct FieldsPerElement{T}
  vals::Vector{T}
end

struct NameMapper{S, I}
  var_ID::NamedTuple{S, Vector{I}}
end

struct StateVec{T, I, NT}
  N_subdomains::I
  N_legendre_poly::I
  var_mapper::NT
  var_names::Tuple{Vararg{Symbol}}
  fields::Vector{FieldsPerElement{T}}
end

"""
```
vars = ((:ρ_0, 1), (:w, 3), (:a, 3), (:α_0, 1))
var_mapper = get_var_mapper(vars)
```
  Returns a NamedTuple containing index
  mapping for each variable. Given the
  example above yields:
    var_mapper[:ρ_0] = [1]
    var_mapper[:w] = [2, 3, 4]
    var_mapper[:a] = [5, 6, 7]
    var_mapper[:α_0] = [8]
"""
function get_var_mapper(vars)
  var_names = tuple([v for (v, nsd) in vars]...)
  end_index = cumsum([nsd for (v, nsd) in vars])
  start_index = [1,[1+x for x in end_index][1:end-1]...]
  vals = [collect(a:b) for (a,b) in zip(start_index, end_index)]
  var_mapper = NamedTuple{var_names}(vals)
  return var_names, var_mapper
end

function StateVec(vars::Tuple{Vararg{Tuple{Symbol, I}}}, grid::Grid{T}, N_legendre_poly::I) where {I, T}
  N_subdomains = max([nsd for (v, nsd) in vars]...)
  n_vars = sum([nsd for (v, nsd) in vars])
  var_names, var_mapper = get_var_mapper(vars)

  # FIXME: Add field elements for N_legendre_poly
  # N_subdomains x N_legendre_poly x N_elements, some vars only have 1 "domain" (averaged over subdomains)
  all_vars = [eltype(grid.n_elem)(0) for v in 1:n_vars]
  fields = [FieldsPerElement(all_vars) for e in over_elems(grid)]
  return StateVec(N_subdomains, N_legendre_poly, var_mapper, var_names, fields)
end

function Base.show(io::IO, sv::StateVec)
  println("N_subdomains    = ", sv.N_subdomains)
  println("N_legendre_poly = ", sv.N_legendre_poly)
  println("var_mapper      = ", sv.var_mapper)
  for fpe in sv.fields
    println(fpe)
  end
  println()
end

over_sub_domains(state_vec::StateVec) = 1:state_vec.N_subdomains
over_sub_domains(state_vec::StateVec, j) = [i for i in 1:state_vec.N_subdomains if !(i==j)]

Base.getindex(sv::StateVec, name::Symbol, e, i_sd=1) = sv.fields[e].vals[sv.var_mapper[name][i_sd]]
function Base.setindex!(sv::StateVec, val, name::Symbol, e, i_sd = 1)
  sv.fields[e].vals[sv.var_mapper[name][i_sd]] = val # requires mutability
end

end