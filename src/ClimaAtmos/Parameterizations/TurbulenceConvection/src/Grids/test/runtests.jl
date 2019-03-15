using Test

using Grids

@testset "Grid init" begin
  res = try
          grid = Grids.Grid(0.0, 1.0, 20)
          true
        catch
          false
        end
  @test res
end

