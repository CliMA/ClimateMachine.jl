module UnitAnnotations

export U, units, V, get_T, value, unit_scale, unit_annotations

using StaticArrays, Unitful
import Unitful: AbstractQuantity, Units

function U end

include("aliases.jl")

"""
Quantity or other scalar type with numeric backing type FT.
"""
const V{FT} = Union{FT, Quantity{FT, D, U}} where {D,U}

"""
    units(FT::Type{T} where {T<:Number}, u::Symbol)

```jldoctest
julia> units(Float64, :velocity)
Unitful.Quantity{Float64,𝐋*𝐓^-1,Unitful.FreeUnits{(m, s^-1),𝐋*𝐓^-1,nothing}}
```
Returns the type variable for the choice of numerical backing type `FT` and unit alias `u`.
"""
function units(FT, u::Symbol)
  units = _unit_alias(Val(u))
  Quantity{FT, dimension(units), typeof(units)}
end

"""
    U(FT::Type{T} where {T<:Number}, u::Unitful.Units)

```jldoctest
julia> U(Float64, u"m*s^-2")
Quantity{Float64,𝐋*𝐓^-2,Unitful.FreeUnits{(m, s^-2),𝐋*𝐓^-2,nothing}}
```

Returns the type variable for the choice of numeric backing type `FT` and preferred units `u`.
"""
units(FT, u::Unitful.Units) = Quantity{FT, dimension(u), typeof(upreferred(u))}

U(FT, u)        = Union{FT, units(FT, u)}
# FIXME
include("uaware.jl")

"""
    unit_scale(::Type{NamedTuple{S, T}} where {S, T<:Tuple}, factor)

Scale the output of a @vars call by the provided units.
"""
@generated function unit_scale(::Type{NamedTuple{S, T}}, factor) where {S, T<:Tuple}
  p(Q,u) = begin
    if Q <: NamedTuple
      return unit_scale(Q, u)
    elseif Q <: SArray
      N = Q.parameters[2]
      return SArray{Q.parameters[1], typeof(oneunit(N)*u), Q.parameters[3], Q.parameters[4]}
    elseif Q <: SHermitianCompact
      N = Q.parameters[2]
      return SHermitianCompact{Q.parameters[1], typeof(oneunit(N)*u), Q.parameters[3]}
    else
      return typeof(oneunit(Q)*u)
    end
  end
  :(return $(NamedTuple{S, Tuple{p.(T.parameters, factor())...}}))
end

@inline Base.:(*)(V::Type{NamedTuple{S, T}}, factor::Units) where {S, T<:Tuple} =
  (unit_scale(V, factor))
@inline Base.:(/)(V::Type{NamedTuple{S, T}}, factor::Units) where {S, T<:Tuple} =
  (unit_scale(V, inv(factor)))

"""
Remove unit annotations, return float in SI units.
"""
value(x::Number) = ustrip(upreferred(x))

"""
Get the numeric backing type for a quantity or numeric scalar type.
"""
get_T(::Type{T}) where {T<:Number} = T
get_T(::Type{T}) where {T<:AbstractQuantity} = Unitful.get_T(T)

Base.Float64(q::Quantity) = unit(q) * Float64(q.val)
Base.Float32(q::Quantity) = unit(q) * Float32(q.val)

end # module
