module TestPrimitivePrognosticConversion

using CLIMAParameters
using CLIMAParameters.Planet
using Test
using StaticArrays
using UnPack

using ClimateMachine
ClimateMachine.init()
const ArrayType = ClimateMachine.array_type()

const clima_dir = dirname(dirname(pathof(ClimateMachine)))

include(joinpath(clima_dir, "test", "Common", "Thermodynamics", "profiles.jl"))
using ClimateMachine.BalanceLaws
using ClimateMachine.ConfigTypes
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Atmos: AtmosModel, DryModel, EquilMoist, NonEquilMoist
using ClimateMachine.BalanceLaws:
    prognostic_to_primitive!, primitive_to_prognostic!
const BL = BalanceLaws

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

atol_temperature = 5e-1
atol_energy = cv_d(param_set) * atol_temperature

import ClimateMachine.BalanceLaws: vars_state

struct TestBL{PS, M} <: BalanceLaw
    param_set::PS
    moisture::M
end

vars_state(bl::TestBL, st::Prognostic, FT) = @vars begin
    ρ::FT
    ρu::SVector{3, FT}
    ρe::FT
    moisture::vars_state(bl.moisture, st, FT)
end

vars_state(bl::TestBL, st::Primitive, FT) = @vars begin
    ρ::FT
    u::SVector{3, FT}
    p::FT
    moisture::vars_state(bl.moisture, st, FT)
end

function assign!(state, bl, nt, st::Prognostic)
    @unpack u, v, w, ρ, e_kin, e_pot, T, q_pt = nt
    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = state.ρ * total_energy(bl.param_set, e_kin, e_pot, T, q_pt)
    assign!(state, bl, bl.moisture, nt, st)
end
assign!(state, bl, moisture::DryModel, nt, ::Prognostic) = nothing
assign!(state, bl, moisture::EquilMoist, nt, ::Prognostic) =
    (state.moisture.ρq_tot = nt.ρ * nt.q_pt.tot)
function assign!(state, bl, moisture::NonEquilMoist, nt, ::Prognostic)
    state.moisture.ρq_tot = nt.ρ * nt.q_pt.tot
    state.moisture.ρq_liq = nt.ρ * nt.q_pt.liq
    state.moisture.ρq_ice = nt.ρ * nt.q_pt.ice
end

function assign!(state, bl, nt, st::Primitive)
    @unpack u, v, w, ρ, p = nt
    state.ρ = ρ
    state.u = SVector(u, v, w)
    state.p = p
    assign!(state, bl, bl.moisture, nt, st)
end
assign!(state, bl, moisture::DryModel, nt, ::Primitive) = nothing
assign!(state, bl, moisture::EquilMoist, nt, ::Primitive) =
    (state.moisture.q_tot = nt.q_pt.tot)
function assign!(state, bl, moisture::NonEquilMoist, nt, ::Primitive)
    state.moisture.q_tot = nt.q_pt.tot
    state.moisture.q_liq = nt.q_pt.liq
    state.moisture.q_ice = nt.q_pt.ice
end


@testset "Prognostic-Primitive conversion (dry)" begin
    FT = Float64
    bl = TestBL(param_set, DryModel())
    vs_prog = vars_state(bl, Prognostic(), FT)
    vs_prim = vars_state(bl, Primitive(), FT)
    prog_arr = zeros(varsize(vs_prog))
    prim_arr = zeros(varsize(vs_prim))
    prog = Vars{vs_prog}(prog_arr)
    prim = Vars{vs_prim}(prim_arr)
    for nt in PhaseDryProfiles(param_set, ArrayType)
        @unpack e_int, e_pot = nt

        # Test prognostic_to_primitive! identity
        assign!(prog, bl, nt, Prognostic())
        prog_0 = deepcopy(parent(prog))
        prim_arr .= 0
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test all(parent(prog) .≈ prog_0)

        # Test primitive_to_prognostic! identity
        assign!(prim, bl, nt, Primitive())
        prim_0 = deepcopy(parent(prim))
        prog_arr .= 0
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test all(parent(prim) .≈ prim_0)
    end
end

@testset "Prognostic-Primitive conversion (EquilMoist)" begin
    FT = Float64
    bl = TestBL(param_set, EquilMoist{FT}())
    vs_prog = vars_state(bl, Prognostic(), FT)
    vs_prim = vars_state(bl, Primitive(), FT)
    prog_arr = zeros(varsize(vs_prog))
    prim_arr = zeros(varsize(vs_prim))
    prog = Vars{vs_prog}(prog_arr)
    prim = Vars{vs_prim}(prim_arr)
    err_max_fwd = 0
    err_max_bwd = 0
    for nt in PhaseEquilProfiles(param_set, ArrayType)
        @unpack e_int, e_pot, q_tot = nt

        # Test prognostic_to_primitive! identity
        assign!(prog, bl, nt, Prognostic())
        prog_0 = deepcopy(parent(prog))
        prim_arr .= 0
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test all(parent(prog)[1:4] .≈ prog_0[1:4])
        @test isapprox(parent(prog)[5], prog_0[5]; atol = atol_energy)
        # @test all(parent(prog)[5] .≈ prog_0[5]) # fails
        @test all(parent(prog)[6] .≈ prog_0[6])
        err_max_fwd = max(abs(parent(prog)[5] .- prog_0[5]), err_max_fwd)

        # Test primitive_to_prognostic! identity
        assign!(prim, bl, nt, Primitive())
        prim_0 = deepcopy(parent(prim))
        prog_arr .= 0
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test all(parent(prim)[1:4] .≈ prim_0[1:4])
        # @test all(parent(prim)[5] .≈ prim_0[5]) # fails
        @test isapprox(parent(prim)[5], prim_0[5]; atol = atol_energy)
        @test all(parent(prim)[6] .≈ prim_0[6])
        err_max_bwd = max(abs(parent(prim)[5] .- prim_0[5]), err_max_bwd)
    end
    # We may want/need to improve this later, so leaving debug info:
    # @show err_max_fwd
    # @show err_max_bwd
end

@testset "Prognostic-Primitive conversion (NonEquilMoist)" begin
    FT = Float64
    bl = TestBL(param_set, NonEquilMoist())
    vs_prog = vars_state(bl, Prognostic(), FT)
    vs_prim = vars_state(bl, Primitive(), FT)
    prog_arr = zeros(varsize(vs_prog))
    prim_arr = zeros(varsize(vs_prim))
    prog = Vars{vs_prog}(prog_arr)
    prim = Vars{vs_prim}(prim_arr)
    for nt in PhaseEquilProfiles(param_set, ArrayType)
        @unpack e_int, e_pot = nt
        # Test prognostic_to_primitive! identity
        assign!(prog, bl, nt, Prognostic())
        prog_0 = deepcopy(parent(prog))
        prim_arr .= 0
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test all(parent(prog) .≈ prog_0)

        # Test primitive_to_prognostic! identity
        assign!(prim, bl, nt, Primitive())
        prim_0 = deepcopy(parent(prim))
        prog_arr .= 0
        primitive_to_prognostic!(bl, bl.moisture, prog, prim, e_pot)
        @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
        prognostic_to_primitive!(bl, bl.moisture, prim, prog, e_int)
        @test all(parent(prim) .≈ prim_0)
    end
end

@testset "Prognostic-Primitive conversion (array interface)" begin
    FT = Float64
    bl = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        moisture = DryModel(),
        init_state_prognostic = x -> x,
    )
    vs_prog = vars_state(bl, Prognostic(), FT)
    vs_prim = vars_state(bl, Primitive(), FT)
    vs_aux = vars_state(bl, Auxiliary(), FT)
    prog_arr = zeros(varsize(vs_prog))
    prim_arr = zeros(varsize(vs_prim))
    aux_arr = zeros(varsize(vs_aux))
    prog = Vars{vs_prog}(prog_arr)
    prim = Vars{vs_prim}(prim_arr)
    aux = Vars{vs_aux}(aux_arr)
    for nt in PhaseDryProfiles(param_set, ArrayType)
        @unpack e_pot = nt

        # Test prognostic_to_primitive! identity
        assign!(prog, bl, nt, Prognostic())
        aux.orientation.Φ = e_pot
        prog_0 = deepcopy(parent(prog))
        prim_arr .= 0
        BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
        @test !all(prog_arr .≈ prim_arr) # ensure not calling fallback
        BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
        @test all(parent(prog) .≈ prog_0)

        # Test primitive_to_prognostic! identity
        assign!(prim, bl, nt, Primitive())
        prim_0 = deepcopy(parent(prim))
        prog_arr .= 0
        BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
        @test !all(prog_arr .≈ prim_arr) # ensure not calling fallback
        BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
        @test all(parent(prim) .≈ prim_0)
    end
end

end
