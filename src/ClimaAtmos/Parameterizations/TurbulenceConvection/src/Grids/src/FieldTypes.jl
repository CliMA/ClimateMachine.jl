module FieldTypes

using ..BoundaryConditions: SingleBC, BCSet

abstract type Field end

struct Center{T<:AbstractFloat} <: Field
  N :: Int
  bcs :: BCSet
  val :: Vector{T}
end

struct Node{T<:AbstractFloat} <: Field
  N :: Int
  bcs :: BCSet
  val :: Vector{T}
end

get_type(f::Node) = Node
get_type(f::Center) = Center

suffix(::Node) = "_n"
suffix(::Center) = "_c"

end