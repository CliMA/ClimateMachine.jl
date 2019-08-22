using CLIMA
using CLIMA.DGmethods
using CLIMA.DGmethods: vars_aux
using CLIMA.Atmos
using CLIMA.Atmos: EquilMoist
using Test

init!() = nothing
source!() = nothing

DF = Float64

@testset "Get variable list" begin
  model = AtmosModel(ConstantViscosityWithDivergence(DF(0)),
                     EquilMoist(),
                     NoRadiation(),
                     source!,
                     NoFluxBC(),
                     init!)

  @test var_names(model, DF) == ["ρ","ρu1","ρu2","ρu3","ρe","moisture_ρq_tot"]
  @test var_names(model, DF, vars_aux) == ["coord1", "coord2", "coord3", "moisture_e_int", "moisture_temperature"]
end
