using Test, StaticArrays, Unitful
using CLIMA.UnitAnnotations
using CLIMA.VariableTemplates
using CLIMA.UnitAnnotations: unit_annotations

# Tests prior to providing unitful information

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
  @vars begin
    ρ::T
    ρu::SVector{3,T}
    ρe::T
    a::state(m.a,T)
    b::state(m.b,T)
    c::state(m.c,T)
    S::SHermitianCompact{3,T,6}
  end
end

state(m::SubModelA, T) = @vars()
state(m::SubModelB, T) = @vars(ρqt::T)
state(m::SubModelC{N}, T) where {N} = @vars(ρk::SVector{N,T})

model = TestModel(SubModelA(), SubModelB(), SubModelC{5}())

st = state(model, Float64)

@test varsize(st) == 17

v = Vars{st}(zeros(MVector{varsize(st),Float64}))
g = Grad{st}(zeros(MMatrix{3,varsize(st),Float64}))

@test v.ρ === 0.0
@test v.ρu === SVector(0.0, 0.0, 0.0)
v.ρu = SVector(1,2,3)
@test v.ρu === SVector(1.0, 2.0, 3.0)

@test v.b.ρqt === 0.0
v.b.ρqt = 12.0
@test v.b.ρqt === 12.0

@test v.S === zeros(SHermitianCompact{3,Float64,6})
v.S = SHermitianCompact{3,Float64,6}(1,2,3,2,3,4,3,4,5)
@test v.S[1,1] === 1.0
@test v.S[1,3] === 3.0
@test v.S[3,1] === 3.0
@test v.S[3,3] === 5.0

v.S = ones(SMatrix{3,3,Int64})
@test v.S[1,1] === 1.0
@test v.S[1,3] === 1.0
@test v.S[3,1] === 1.0
@test v.S[3,3] === 1.0

@test propertynames(v.a) == ()
@test propertynames(g.a) == ()

@test g.ρu == zeros(SMatrix{3,3,Float64})
g.ρu = SMatrix{3,3}(1:9)
@test g.ρu == SMatrix{3,3,Float64}(1:9)

@test size(v.c.ρk) == (5,)
@test size(g.c.ρk) == (3,5)

sv = similar(v)
@test typeof(sv) == typeof(v)
@test size(parent(sv)) == size(parent(v))

sg = similar(g)
@test typeof(sg) == typeof(g)
@test size(parent(sg)) == size(parent(g))

@test flattenednames(st) == ["ρ","ρu[1]","ρu[2]","ρu[3]","ρe",
                            "b.ρqt",
                            "c.ρk[1]","c.ρk[2]","c.ρk[3]","c.ρk[4]","c.ρk[5]",
                            "S[1,1]", "S[2,1]", "S[3,1]", "S[2,2]", "S[3,2]", "S[3,3]"]

# With unit assignment

function unitful_state(m::TestModel, T)
  @vars begin
    ρ::units(T,:density)
    ρu::SVector{3,units(T,:massflux)}
    ρe::units(T,:energypv)
    a::unitful_state(m.a,T)
    b::unitful_state(m.b,T)
    c::unitful_state(m.c,T)
    S::SHermitianCompact{3,units(T,:kinvisc),6}
  end
end

unitful_state(m::SubModelA, T) = @vars()
unitful_state(m::SubModelB, T) = @vars(ρqt::units(T,:density))
unitful_state(m::SubModelC{N}, T) where {N} = @vars(ρk::SVector{N,units(T,:density)})

ust = unitful_state(model, Float64)
@test varsize(ust) === varsize(st)

vu = Vars{ust}(zeros(MVector{varsize(ust), Float64}))
gu = Grad{ust}(zeros(MVector{varsize(ust), Float64}))

vu.ρ = 1.0
@test vu.ρ == 1.0u"kg/m^3"
vu.ρ = 2.0u"kg/m^3"
@test vu.ρ == 2.0u"kg/m^3"

ust_scaled_nop = unit_scale(ust, NoUnits)
vusn = Vars{ust_scaled_nop}(zeros(MVector{varsize(ust), Float64}))
@test ust_scaled_nop === ust
vusm = Vars{unit_scale(ust, u"m")}(zeros(MVector{varsize(ust), Float64}))
@test typeof(vusm.ρ) === typeof(vu.ρ * u"m")
vu.S = @SVector[0.0 for i in 1:6]
@test typeof(vu.S) === SHermitianCompact{3, units(Float64, u"m^2/s"),6}
vusn.S = @SVector[0.0u"m^2/s" for i in 1:6]
@test typeof(vusn.S) === SHermitianCompact{3, units(Float64, u"m^2/s"),6}
@test vu.S === vusn.S

@test flattenednames(ust) == ["ρ","ρu[1]","ρu[2]","ρu[3]","ρe",
                            "b.ρqt",
                            "c.ρk[1]","c.ρk[2]","c.ρk[3]","c.ρk[4]","c.ρk[5]",
                            "S[1,1]", "S[2,1]", "S[3,1]", "S[2,2]", "S[3,2]", "S[3,3]"]

# Test unit detection
@test unit_annotations(vu)
@test !unit_annotations(v)
@test unit_annotations(gu)
@test !unit_annotations(g)
