using Test
using ClimateMachine.BalanceLaws

import ClimateMachine.BalanceLaws: eq_tends, prognostic_vars

struct TestBL <: BalanceLaw end
struct X <: PrognosticVariable end
struct Y <: PrognosticVariable end
struct F1{PV} <: TendencyDef{Flux{FirstOrder}, PV} end
struct F2{PV} <: TendencyDef{Flux{SecondOrder}, PV} end
struct S{PV} <: TendencyDef{Source, PV} end

prognostic_vars(::TestBL) = (X(), Y())
eq_tends(::X, ::TestBL, ::Flux{FirstOrder}) = (F1{X}(),)
eq_tends(::Y, ::TestBL, ::Flux{FirstOrder}) = (F1{Y}(),)
eq_tends(::X, ::TestBL, ::Flux{SecondOrder}) = (F2{X}(),)
eq_tends(::Y, ::TestBL, ::Flux{SecondOrder}) = (F2{Y}(),)
eq_tends(::X, ::TestBL, ::Source) = (S{X}(),)
eq_tends(::Y, ::TestBL, ::Source) = (S{Y}(),)

@testset "BalanceLaws" begin
    bl = TestBL()
    @test prognostic_vars(bl) == (X(), Y())
    @test fluxes(bl, FirstOrder()) == (F1{X}(), F1{Y}())
    @test fluxes(bl, SecondOrder()) == (F2{X}(), F2{Y}())
    @test sources(bl) == (S{X}(), S{Y}())
    show_tendencies(bl)
    show_tendencies(bl; include_params = true)
end
