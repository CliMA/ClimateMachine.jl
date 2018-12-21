using Test
using PlanetParameters

function main()
  println()
  println("Testing: ",relpath(@__FILE__))
  @testset begin
    # Not sure how to really test this...
    @test @isdefined R_d
    @test L_f0 == L_s0 - L_v0
  end
end

main()
