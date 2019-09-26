using Test, Printf

using CLIMA.TurbulenceConvection.FiniteDifferenceGrids
using CLIMA.TurbulenceConvection.StateVecs

n_elems_real = 10 # number of elements

grid = Grid(0.0, 1.0, n_elems_real)
vars = ( (:ρ_0, DomainSubSet(gm=true)),
         (:a,   DomainSubSet(gm=true,en=true,ud=true)),
         (:w,   DomainSubSet(gm=true,en=true,ud=true)) )
dd = DomainDecomp(gm=1,en=1,ud=2)
state_vec = StateVec(vars, grid, dd)
idx = DomainIdx(state_vec)
i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)

@testset "Memory access" begin
  state_vec[:ρ_0, i_gm] = 2.0
  @test state_vec[:ρ_0, i_gm] == 2.0

  state_vec[:w, 1, i_gm] = 3.0
  @test state_vec[:w, 1, i_gm] == 3.0

  @test_throws BoundsError state_vec[:w, 1, 1000] = 3.0
  @test_throws BoundsError state_vec[:ρ_0, 1, i_en] = 3.0

  for k in over_elems(grid)
    ρ_0_e = state_vec[:ρ_0, k]
    for i in alldomains(idx)
      w_0_e_i = state_vec[:w, k, i]
    end
  end
  sprint(show, state_vec)
end

@testset "Boundary conditions" begin
  state_vec[:ρ_0, 1] = 0
  state_vec[:ρ_0, 2] = 0
  state_vec[:ρ_0, 1] = 0
  state_vec[:ρ_0, 2] = 0
  k = 1
  apply_Dirichlet!(state_vec, :ρ_0, grid, 2, Zmin())
  @test state_vec[:ρ_0, k] ≈ 4
  apply_Neumann!(state_vec, :ρ_0, grid, -2/grid.Δz, Zmin())
  @test state_vec[:ρ_0, k] ≈ 2

  k = grid.n_elem
  apply_Dirichlet!(state_vec, :ρ_0, grid, 2, Zmax())
  @test state_vec[:ρ_0, k] ≈ 4
  apply_Neumann!(state_vec, :ρ_0, grid,  2/grid.Δz, Zmax())
  @test state_vec[:ρ_0, k] ≈ 2
end

@testset "Cuts" begin
  state_vec[:ρ_0, 1] = 1.0
  state_vec[:ρ_0, 2] = 2.0
  state_vec[:ρ_0, 3] = 3.0
  state_vec[:a, 1] = 1.0
  state_vec[:a, 2] = 2.0
  state_vec[:a, 3] = 3.0
  ρα_0_cut = state_vec[:ρ_0, Cut(2)].*state_vec[:a, Cut(2)]
  @test all(ρα_0_cut .== ρα_0_cut)
  ρα_0_dual = state_vec[:ρ_0, Dual(2)].*state_vec[:a, Dual(2)]
  @test all(ρα_0_dual .== ρα_0_dual)
end

@testset "Auxiliary" begin
  state_vec[:a, 3] = 1
  @test !isnan(state_vec)
  @test !isinf(state_vec)
  state_vec[:a, 3] = 1/0
  @test isinf(state_vec)
  state_vec[:a, 3] = NaN
  @test isnan(state_vec)
end
