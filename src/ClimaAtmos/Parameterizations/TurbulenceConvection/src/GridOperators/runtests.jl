using Test

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.GridOperators

@testset "GridOperators" begin
  n_elems_real = 10
  grid = Grids.Grid(0.0, 1.0, n_elems_real)
  f = [2, 2, 3]
  @test ∇_z(f, grid) ≈ (f[1] + f[3] - 2*f[2])*grid.dzi
  @test Δ_z(f, grid) ≈ (f[3] - 2*f[2] + f[1])*grid.dzi2
  u = [2, 3, 4]
  @test adv_upwind(f, u, grid) ≈ u[ 2 ]*(f[2] - f[1]) * grid.dzi
  u = [2, -2, 4]
  @test adv_upwind(f, u, grid) ≈ u[ 2 ]*(f[3] - f[2]) * grid.dzi
end

