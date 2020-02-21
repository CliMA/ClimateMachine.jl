module VarsArrays
using ..VariableTemplates: varseltype, varsize, varsindex, Vars

export VarsArray

"""
    VarsArray <: AbstractArray

`VarsArray`s are Julia arrays that know about know about `vars` along a given
dimension
"""
struct VarsArray{V, dim, T, N1, N, AT<:AbstractArray{T, N}} <: AbstractArray{V, N1}
  array::AT
end
Base.parent(A::VarsArray) = A.array

"""
    VarsArray{V, dim}(A)

Constructs a `VarsArray` from `A`. The vars dimension is `dim` and the names of
the vars `V` are a `NamedTuple` generated in a manner consistent with the
`@vars` macro from `VariableTemplates`

# Examples
```julia-repl
using StaticArrays
using CLIMA.VariableTemplates
using CLIMA.VarsArrays

FT = Float64

W = @vars begin
  a::FT
  b::SVector{3, FT}
end

V = @vars begin
  a::FT
  b::SVector{3, FT}
  c::SMatrix{3,3,FT, 9}
  e::W
end
N = varsize(V)

A1 = VarsArray{V, 1}(rand(N, 1, 2))

A1.a
A1.e
A1.e.a
```
"""
function VarsArray{V, dim}(A::AT) where {V, dim, T, N, AT<:AbstractArray{T, N}}
  @assert varseltype(V) <: T
  @assert size(A, dim) == varsize(V)
  N1 = N-1
  return VarsArray{V, dim, T, N1, N, AT}(A)
end

# Convience functions for type space info
getvars(A::VarsArray{V}) where V = V
getdim(A::VarsArray{V, dim }) where {V, dim} = dim

function Base.getproperty(A::VarsArray, sym::Symbol)
  V = getvars(A)
  if sym ∈ fieldnames(V)
    N = ndims(A.array)
    dim = getdim(A)
    varrange = varsindex(V, sym)
    T = fieldtype(V, sym)
    Av = view(A.array, ntuple(i->i==dim ? varrange : Colon(), N)...)
    if !(T<:NamedTuple)
      T = NamedTuple{(sym,), Tuple{T,}}
    end
    return VarsArray{T, dim}(Av)
  else
    return getfield(A, sym)
  end
end

# TODO: How to inherit all these properties?
function Base.size(A::VarsArray)
  dim = getdim(A)
  sz = size(A.array)
  sz = (sz[1:dim-1]..., sz[dim+1:end]...)
end


function Base.getindex(A::VarsArray, CI::CartesianIndex)
  dim = getdim(A)
  V = getvars(A)
  sz = size(A)
  preI = CartesianIndex(CI.I[1:dim-1])
  postI = CartesianIndex(CI.I[dim:end])
  # If it's only one field, then just return what vars would
  # otherwise return a vars
  if length(fieldnames(V)) == 1
    getproperty(Vars{V}(A.array[preI, :, postI]), fieldnames(V)[1])
  else
    Vars{V}(A.array[preI, :, postI])
  end
end
function Base.getindex(A::VarsArray, i::Int)
  dim = getdim(A)
  V = getvars(A)
  sz = size(A)
  CI = CartesianIndices(sz)[i]
  getindex(A, CI)
end
function Base.getindex(A::VarsArray{V, dim, T, N}, i::Vararg{Int, N}) where {V, dim, T, N}
  A[CartesianIndex(i)]
end

# TODO: Figure out setindex
#    - Want A.a[1] = 1
#    - Want A[1] .= 1
#    - Want A[1].a = 1

# TODO: Fix views

end
