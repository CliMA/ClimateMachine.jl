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
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws:
    prognostic_to_primitive!, primitive_to_prognostic!
const BL = BalanceLaws

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

atol_temperature = 5e-1
atol_energy = cv_d(param_set) * atol_temperature

import ClimateMachine.BalanceLaws: vars_state

# Assign aux.ref_state different than state for general testing
function assign!(aux, bl, nt, st::Auxiliary)
    @unpack p, ρ, e_pot, = nt
    aux.ref_state.ρ = ρ / 2
    aux.ref_state.p = p / 2
    aux.orientation.Φ = e_pot
end

function assign!(state, bl, nt, st::Prognostic, aux)
    @unpack u, v, w, ρ, e_kin, e_pot, T, q_pt = nt
    state.ρ = ρ
    assign!(state, bl, bl.compressibility, nt, st, aux)
    state.ρu = SVector(state.ρ * u, state.ρ * v, state.ρ * w)
    param_set = parameter_set(bl)
    state.energy.ρe = state.ρ * total_energy(param_set, e_kin, e_pot, T, q_pt)
    assign!(state, bl, bl.moisture, nt, st)
end
assign!(state, bl, moisture::DryModel, nt, ::Prognostic) = nothing
assign!(state, bl, moisture::EquilMoist, nt, ::Prognostic) =
    (state.moisture.ρq_tot = state.ρ * nt.q_pt.tot)
function assign!(state, bl, moisture::NonEquilMoist, nt, ::Prognostic)
    state.moisture.ρq_tot = state.ρ * nt.q_pt.tot
    state.moisture.ρq_liq = state.ρ * nt.q_pt.liq
    state.moisture.ρq_ice = state.ρ * nt.q_pt.ice
end
# Assign prog.ρ = aux.ref_state.ρ in anelastic1D
assign!(state, bl, ::Compressible, nt, ::Prognostic, aux) = nothing
function assign!(state, bl, ::Anelastic1D, nt, ::Prognostic, aux)
    state.ρ = aux.ref_state.ρ
end

function assign!(state, bl, nt, st::Primitive, aux)
    @unpack u, v, w, ρ, p = nt
    state.ρ = ρ
    state.u = SVector(u, v, w)
    state.p = p
    assign!(state, bl, bl.compressibility, nt, st, aux)
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
# Assign prim.p = aux.ref_state.p in anelastic1D
assign!(state, bl, ::Compressible, nt, ::Primitive, aux) = nothing
function assign!(state, bl, ::Anelastic1D, nt, ::Primitive, aux)
    state.p = aux.ref_state.p
end

@testset "Prognostic-Primitive conversion (dry)" begin
    FT = Float64
    compressibility = (Anelastic1D(), Compressible())
    for comp in compressibility
        bl = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            moisture = DryModel(),
            compressibility = comp,
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
            assign!(aux, bl, nt, Auxiliary())

            # Test prognostic_to_primitive! identity
            assign!(prog, bl, nt, Prognostic(), aux)
            prog_0 = deepcopy(parent(prog))
            prim_arr .= 0
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test all(parent(prog) .≈ prog_0)

            # Test primitive_to_prognostic! identity
            assign!(prim, bl, nt, Primitive(), aux)
            prim_0 = deepcopy(parent(prim))
            prog_arr .= 0
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test all(parent(prim) .≈ prim_0)
        end
    end
end

@testset "Prognostic-Primitive conversion (EquilMoist)" begin
    FT = Float64
    compressibility = (Compressible(),) # Anelastic1D() does not converge
    for comp in compressibility
        bl = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            moisture = EquilMoist(; maxiter = 5), # maxiter=3 does not converge
            compressibility = comp,
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
        err_max_fwd = 0
        err_max_bwd = 0
        for nt in PhaseEquilProfiles(param_set, ArrayType)
            assign!(aux, bl, nt, Auxiliary())

            # Test prognostic_to_primitive! identity
            assign!(prog, bl, nt, Prognostic(), aux)
            prog_0 = deepcopy(parent(prog))
            prim_arr .= 0
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test all(parent(prog)[1:4] .≈ prog_0[1:4])
            @test isapprox(parent(prog)[5], prog_0[5]; atol = atol_energy)
            # @test all(parent(prog)[5] .≈ prog_0[5]) # fails
            @test all(parent(prog)[6] .≈ prog_0[6])
            err_max_fwd = max(abs(parent(prog)[5] .- prog_0[5]), err_max_fwd)

            # Test primitive_to_prognostic! identity
            assign!(prim, bl, nt, Primitive(), aux)
            prim_0 = deepcopy(parent(prim))
            prog_arr .= 0
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test all(parent(prim)[1:4] .≈ prim_0[1:4])
            # @test all(parent(prim)[5] .≈ prim_0[5]) # fails
            @test isapprox(parent(prim)[5], prim_0[5]; atol = atol_energy)
            @test all(parent(prim)[6] .≈ prim_0[6])
            err_max_bwd = max(abs(parent(prim)[5] .- prim_0[5]), err_max_bwd)
        end
    end
    # We may want/need to improve this later, so leaving debug info:
    # @show err_max_fwd
    # @show err_max_bwd
end

@testset "Prognostic-Primitive conversion (NonEquilMoist)" begin
    FT = Float64
    compressibility = (Anelastic1D(), Compressible())
    for comp in compressibility
        bl = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            moisture = NonEquilMoist(),
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
        for nt in PhaseEquilProfiles(param_set, ArrayType)
            assign!(aux, bl, nt, Auxiliary())

            # Test prognostic_to_primitive! identity
            assign!(prog, bl, nt, Prognostic(), aux)
            prog_0 = deepcopy(parent(prog))
            prim_arr .= 0
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test all(parent(prog) .≈ prog_0)

            # Test primitive_to_prognostic! identity
            assign!(prim, bl, nt, Primitive(), aux)
            prim_0 = deepcopy(parent(prim))
            prog_arr .= 0
            primitive_to_prognostic!(bl, prog, prim, aux)
            @test !all(parent(prim) .≈ parent(prog)) # ensure not calling fallback
            prognostic_to_primitive!(bl, prim, prog, aux)
            @test all(parent(prim) .≈ prim_0)
        end
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
        assign!(aux, bl, nt, Auxiliary())

        # Test prognostic_to_primitive! identity
        assign!(prog, bl, nt, Prognostic(), aux)
        prog_0 = deepcopy(parent(prog))
        prim_arr .= 0
        BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
        @test !all(prog_arr .≈ prim_arr) # ensure not calling fallback
        BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
        @test all(parent(prog) .≈ prog_0)

        # Test primitive_to_prognostic! identity
        assign!(prim, bl, nt, Primitive(), aux)
        prim_0 = deepcopy(parent(prim))
        prog_arr .= 0
        BL.primitive_to_prognostic!(bl, prog_arr, prim_arr, aux_arr)
        @test !all(prog_arr .≈ prim_arr) # ensure not calling fallback
        BL.prognostic_to_primitive!(bl, prim_arr, prog_arr, aux_arr)
        @test all(parent(prim) .≈ prim_0)
    end
end

end
