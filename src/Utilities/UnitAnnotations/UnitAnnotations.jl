module UnitAnnotations

export U, V, get_T, value, unit_scale,
       space_unit, mass_unit, time_unit, temp_unit

using StaticArrays, Unitful; using CLIMA.UnitAnnotations #FIXME
import Unitful: AbstractQuantity

include("aliases.jl")

"""
Quantity or other scalar type with numeric backing type FT.
"""
V{FT} = Union{FT, Quantity{FT, D, U}} where {D,U}

"""
    U(FT::Type{T} where {T<:Number}, u::Symbol)

```jldoctest
julia> U(Float64, :vel)
Union{Float64, Unitful.Quantity{Float64,𝐋*𝐓^-1,Unitful.FreeUnits{(m, s^-1),𝐋*𝐓^-1,nothing}}}
```
Returns the type variable for the choice of numerical backing type `FT` and unit alias `u`.
"""
function U(FT, u::Symbol)
  units = upreferred(unit_glossary[u])
  Union{FT, Quantity{FT, dimension(units), typeof(units)}}
end

"""
    U(FT::Type{T} where {T<:Number}, u::Unitful.Units)

```jldoctest
julia> U(Float64, u"m*s^-2")
Quantity{Float64,𝐋*𝐓^-2,Unitful.FreeUnits{(m, s^-2),𝐋*𝐓^-2,nothing}}
```

Returns the type variable for the choice of numeric backing type `FT` and preferred units `u`.
"""
U(FT, u::Unitful.Units) = Quantity{FT, dimension(u), typeof(upreferred(u))}

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

"""
Remove unit annotations, return float in SI units.
"""
value(x::Number) = ustrip(upreferred(x))

"""
Get the numeric backing type for a quantity or numeric scalar type.
"""
get_T(::Type{T}) where {T<:Number} = T
get_T(::Type{T}) where {T<:AbstractQuantity} = Unitful.get_T(T)

end # module
