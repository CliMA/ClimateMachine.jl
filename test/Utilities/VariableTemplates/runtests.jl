using Test, StaticArrays
using CLIMA.VariableTemplates

struct TestModel{A,B,C}
  a::A
  b::B
  c::C
end

struct SubModelA
end
struct SubModelB
end
struct SubModelC{N}
end

function state(m::TestModel, T)
  NamedTuple{(:ρ, :ρu, :ρe, :a, :b, :c), 
  Tuple{T, SVector{3,T}, T, state(m.a,T), state(m.b,T), state(m.c, T)}}
end

state(m::SubModelA, T) = Tuple{}
state(m::SubModelB, T) = NamedTuple{(:ρqt,), Tuple{T}}
state(m::SubModelC{N}, T) where {N} = NamedTuple{(:ρk,), Tuple{SVector{N,T}}}

model = TestModel(SubModelA(), SubModelB(), SubModelC{5}())

st = state(model, Float64)

@test varsize(st) == 11

v = Vars{st}(zeros(MVector{11,Float64}))
g = Grad{st}(zeros(MMatrix{3,11,Float64}))

@test v.ρ === 0.0
@test v.ρu === SVector(0.0, 0.0, 0.0)
v.ρu = SVector(1,2,3)
@test v.ρu === SVector(1.0, 2.0, 3.0)

@test v.b.ρqt === 0.0
v.b.ρqt = 12.0
@test v.b.ρqt === 12.0

@test propertynames(v.a) == ()
@test propertynames(g.a) == ()

@test g.ρu == zeros(SMatrix{3,3,Float64})
g.ρu = SMatrix{3,3}(1:9)
@test g.ρu == SMatrix{3,3,Float64}(1:9)


@test size(v.c.ρk) == (5,)
@test size(g.c.ρk) == (3,5)

