using Test
using CLIMA.PlanetParameters

@testset begin
  # Not sure how to really test this...
  @test @isdefined R_d
  @test LH_f0 == LH_s0 - LH_v0
end
