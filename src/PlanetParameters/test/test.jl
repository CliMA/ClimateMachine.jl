using Test
using PlanetParameters

function main()
  println()
  println("Testing: ",relpath(@__FILE__))
  @testset begin
    # Not sure how to really test this...
    @test @isdefined R_d
    @test LH_f0 == LH_s0 - LH_v0
  end
end

main()
