using Test
using Random
using StaticArrays: SVector
Random.seed!(1234)

using ClimateMachine.VariableTemplates: @vars, varsize, Vars
using ClimateMachine.BalanceLaws
const BL = BalanceLaws
import ClimateMachine.BalanceLaws: vars_state, eq_tends, prognostic_vars

struct TestBL <: BalanceLaw end
struct X <: AbstractPrognosticVariable end
struct Y <: AbstractPrognosticVariable end
struct F1 <: TendencyDef{Flux{FirstOrder}} end
struct F2 <: TendencyDef{Flux{SecondOrder}} end
struct S <: TendencyDef{Source} end

prognostic_vars(::TestBL) = (X(), Y())
eq_tends(::X, ::TestBL, ::Flux{FirstOrder}) = (F1(),)
eq_tends(::Y, ::TestBL, ::Flux{FirstOrder}) = (F1(),)
eq_tends(::X, ::TestBL, ::Flux{SecondOrder}) = (F2(),)
eq_tends(::Y, ::TestBL, ::Flux{SecondOrder}) = (F2(),)
eq_tends(::X, ::TestBL, ::Source) = (S(),)
eq_tends(::Y, ::TestBL, ::Source) = (S(),)

@testset "BalanceLaws" begin
    bl = TestBL()
    @test prognostic_vars(bl) == (X(), Y())
    show_tendencies(bl)
    show_tendencies(bl; include_module = true)
    show_tendencies(bl; table_complete = true)
end

vars_state(bl::TestBL, st::Prognostic, FT) = @vars begin
    ρ::FT
    ρu::SVector{3, FT}
end

vars_state(bl::TestBL, st::Auxiliary, FT) = @vars()

@testset "Prognostic-Primitive conversion (identity)" begin
    FT = Float64
    bl = TestBL()
    vs_prog = vars_state(bl, Prognostic(), FT)
    vs_prim = vars_state(bl, Primitive(), FT)
    vs_aux = vars_state(bl, Auxiliary(), FT)
    prim_arr = zeros(varsize(vs_prim))
    prog_arr = zeros(varsize(vs_prog))
    aux_arr = zeros(varsize(vs_aux))

    # Test prognostic_to_primitive! identity
    prog_arr .= rand(varsize(vs_prog))
    prog_0 = deepcopy(prog_arr)
    prim_arr .= 0
    BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
    BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
    @test all(prog_arr .≈ prog_0)

    # Test primitive_to_prognostic! identity
    prim_arr .= rand(varsize(vs_prim))
    prim_0 = deepcopy(prim_arr)
    prog_arr .= 0
    BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
    BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
    @test all(prim_arr .≈ prim_0)
end
