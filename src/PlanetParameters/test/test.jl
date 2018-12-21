using Test
using PlanetParameters

function main()
  println()
  println("Testing: ",relpath(@__FILE__))
  @testset begin
    # Not sure how to really test this...
    @test @isdefined R_d
    @test Float64(kappa_d) == Float64(2//7)
  end
end

main()
