using Test, StaticArrays
using ClimateMachine.VariableTemplates: varsindex, @vars, varsindices
using StaticArrays

struct TestMoistureModel{FT} end
struct TestAtmosModel{FT}
    moisture::TestMoistureModel{FT}
end

function vars_state(::TestMoistureModel, FT)
    @vars begin
        ρq_tot::FT
        ρq_x::SVector{5, FT}
        ρq_liq::FT
        ρq_vap::FT
    end
end
function vars_state(m::TestAtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        moisture::vars_state(m.moisture, FT)
    end
end
@testset "Varsindex" begin
    FT = Float64
    m = TestAtmosModel(TestMoistureModel{FT}())

    @test 1:1 === varsindex(vars_state(m, FT), :ρ)
    @test 2:4 === varsindex(vars_state(m, FT), :ρu)
    @test 5:5 === varsindex(vars_state(m, FT), :ρe)

    # Since moisture is defined recusively this will get all the fields
    moist = varsindex(vars_state(m, FT), :moisture)
    @test 6:13 === moist

    # To get the specific ones we can do
    @test 6:6 === varsindex(vars_state(m, FT), :moisture, :ρq_tot)
    @test 7:11 === varsindex(vars_state(m, FT), :moisture, :ρq_x)
    @test 12:12 === varsindex(vars_state(m, FT), :moisture, :ρq_liq)
    @test 13:13 === varsindex(vars_state(m, FT), :moisture, :ρq_vap)
    # or
    @test 6:6 === moist[varsindex(vars_state(m.moisture, FT), :ρq_tot)]
    @test 7:11 === moist[varsindex(vars_state(m.moisture, FT), :ρq_x)]
    @test 12:12 === moist[varsindex(vars_state(m.moisture, FT), :ρq_liq)]
    @test 13:13 === moist[varsindex(vars_state(m.moisture, FT), :ρq_vap)]

    @test (1,) === varsindices(vars_state(m, FT), :ρ)
    @test (2, 3, 4) === varsindices(vars_state(m, FT), :ρu)
    @test (1, 5) === varsindices(vars_state(m, FT), :ρ, :ρe)
    @test (12,) === varsindices(vars_state(m, FT), :(moisture.ρq_liq))
    let
        vars = ("ρe", "moisture.ρq_x", "moisture.ρq_vap")
        @test (5, 7, 8, 9, 10, 11, 13) === varsindices(vars_state(m, FT), vars)
    end
end
