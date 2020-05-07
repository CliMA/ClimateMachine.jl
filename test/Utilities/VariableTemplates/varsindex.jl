using Test, StaticArrays
using ClimateMachine.VariableTemplates: varsindex, @vars
using StaticArrays

struct MoistureModel{FT} end
struct AtmosModel{FT}
    moisture::MoistureModel{FT}
end

function vars_state(::MoistureModel, FT)
    @vars begin
        ρq_tot::FT
        ρq_x::SVector{5, FT}
        ρq_liq::FT
        ρq_vap::FT
    end
end
function vars_state(m::AtmosModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        moisture::vars_state(m.moisture, FT)
    end
end
@testset "Varsindex" begin
    FT = Float64
    m = AtmosModel(MoistureModel{FT}())

    @test 1:1 === varsindex(vars_state(m, FT), :ρ)
    @test 2:4 === varsindex(vars_state(m, FT), :ρu)
    @test 5:5 === varsindex(vars_state(m, FT), :ρe)

    # Since moisture is defined recusively this will get all the fields
    moist = varsindex(vars_state(m, FT), :moisture)
    @test 6:13 === moist

    # To get the specific onnes we need something like
    @test 6:6 === moist[varsindex(vars_state(m.moisture, FT), :ρq_tot)]
    @test 7:11 === moist[varsindex(vars_state(m.moisture, FT), :ρq_x)]
    @test 12:12 === moist[varsindex(vars_state(m.moisture, FT), :ρq_liq)]
    @test 13:13 === moist[varsindex(vars_state(m.moisture, FT), :ρq_vap)]
end
