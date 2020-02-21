using StaticArrays
using CLIMA.VariableTemplates
using CLIMA.VarsArrays

FT = Float64

W = @vars begin
  a::FT
  b::SVector{4, FT}
end

V = @vars begin
  a::FT
  b::SVector{3, FT}
  c::SMatrix{3,8,FT, 24}
  d::FT
  e::W
end
@show N = varsize(V)

@show varseltype(V) <: FT
A1 = VarsArray{V, 1}(rand(N, 3, 4))
# A2 = VarsArray{V, 2}(rand(1, N, 2))
# A3 = VarsArray{V, 3}(rand(4, 10, N))
# @show A1.a == view(A1.array, 1:1, :, :)
# @show A2.a == view(A2.array, :, 1:1, :)
# @show A3.a == view(A3.array, :, :, 1:1)
# @show A1.b == view(A1.array, 2:4, :, :)
# @show A2.b == view(A2.array, :, 2:4, :)
# @show A3.b == view(A3.array, :, :, 2:4)
# @show A1.e == view(A1.array, 30:34, :, :)
# @show A2.e == view(A2.array, :, 30:34, :)
# @show A3.e == view(A3.array, :, :, 30:34)
# @show eltype(A1) == Float64
# @show eltype(A2) == Float64
# @show eltype(A3) == Float64
nothing
