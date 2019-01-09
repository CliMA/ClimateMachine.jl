
using Utilities

using Test

using Utilities.MoistThermodynamics, PlanetParameters

@testset "moist thermodynamics" begin
  @test air_pressure([1, 1, 1], [1, 1, 2], [1, 0, 1], [0, 0, 0.5], [0, 0, 0]) ≈ [R_v, R_d, R_v] 
  @test gas_constant_moist([0, 1, 0.5], [0, 0, 0.5], [0, 0, 0]) ≈ [R_d, R_v, R_d/2]
  @test cp_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d, cp_v, cp_l, cp_i]
  @test cv_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d - R_d, cp_v - R_v, cv_l, cv_i]
  @test air_temperature(cv_d*(300-T_0).+[0, 10, 10], [0, 10, 0], [0, 0, 10], 0, 0, 0) ≈ [300, 300, 300]
  qt=0.23; @test air_temperature(cv_m([0, qt], 0, 0).*(300-T_0).+[0, qt*IE_v0], 0, 0, [0, qt], 0, 0) ≈ [300, 300]
  @test latent_heat_vapor(T_0) ≈ LH_v0
  @test latent_heat_fusion(T_0) ≈ LH_f0
  @test latent_heat_sublim(T_0) ≈ LH_s0
  @test sat_vapor_press_liquid(T_triple) ≈ press_triple
  @test sat_vapor_press_ice(T_triple) ≈ press_triple
end
