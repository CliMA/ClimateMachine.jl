using Test

using CLIMA.TurbulenceConvection.Grids

@testset "Grid interface" begin
  for n_ghost in (1, 2)
    n_elems = 12
    n_elems_real = n_elems-2*n_ghost
    elem_indexes = 1:n_elems
    elem_indexes_real = elem_indexes[1+n_ghost:end-n_ghost]
    Δz = 1/n_elems_real
    grid = Grids.Grid(0.0, 1.0, n_elems_real, n_ghost)
    @test all(over_elems(grid) .== elem_indexes)
    @test all(over_elems_real(grid) .== elem_indexes_real)
    @test length(over_elems(grid)) == n_elems
    @test length(over_elems_real(grid)) == n_elems_real
    @test first_cell_above_surface(grid) == 1+n_ghost
    @test get_z(grid, first_cell_above_surface(grid)) ≈ grid.zn_surf + Δz/2
    @test over_elems_ghost(grid) == [(1:n_ghost)..., (n_elems+1-n_ghost:n_elems)...]
    @test grid.zn_surf ≈ 0.0
    @test grid.zn_top  ≈ 1.0
    @test grid.zc_surf ≈ grid.zn_surf + Δz/2
    @test grid.zc_top  ≈ grid.zn_top  - Δz/2
  end

end
