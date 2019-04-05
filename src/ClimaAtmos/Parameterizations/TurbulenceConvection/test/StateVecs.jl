using Test, Printf

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.GridOperators
using CLIMA.TurbulenceConvection.BoundaryConditions

n_subdomains = 3 # number of sub-domains
n_elems_real = 10 # number of elements

grid = Grid(0.0, 1.0, n_elems_real)
vars = ( (:ρ_0, 1), (:a, n_subdomains), (:w, n_subdomains) )
state_vec = StateVec(vars, grid)

@testset "Memory access" begin
  state_vec[:ρ_0, 1] = 2.0
  @test state_vec[:ρ_0, 1] == 2.0

  state_vec[:w, 1, 1] = 3.0
  @test state_vec[:w, 1, 1] == 3.0

  @test_throws BoundsError state_vec[:w, 1, 4] = 3.0
  @test_throws BoundsError state_vec[:ρ_0, 1, 2] = 3.0

  @test over_sub_domains(state_vec) == 1:3
  @test over_sub_domains(state_vec, 2) == [1,3]
  @test over_sub_domains(state_vec, :ρ_0) == 1:1
  @test over_sub_domains(state_vec, :a) == 1:3

  for k in over_elems(grid)
    ρ_0_e = state_vec[:ρ_0, k]
    for i in over_sub_domains(state_vec)
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
  Dirichlet!(state_vec, :ρ_0, 2, grid, Bottom())
  @test state_vec[:ρ_0, k] ≈ 4
  Neumann!(state_vec, :ρ_0, -2/grid.dz, grid, Bottom())
  @test state_vec[:ρ_0, k] ≈ 2

  k = grid.n_elem
  Dirichlet!(state_vec, :ρ_0, 2, grid, Top())
  @test state_vec[:ρ_0, k] ≈ 4
  Neumann!(state_vec, :ρ_0, 2/grid.dz, grid, Top())
  @test state_vec[:ρ_0, k] ≈ 2
end

@testset "Slices" begin
  state_vec[:ρ_0, 1] = 1.0
  state_vec[:ρ_0, 2] = 2.0
  state_vec[:ρ_0, 3] = 3.0
  state_vec[:a, 1] = 1.0
  state_vec[:a, 2] = 2.0
  state_vec[:a, 3] = 3.0
  ρα_0_slice = state_vec[:ρ_0, Slice(2)].*state_vec[:a, Slice(2)]
  @test all(ρα_0_slice .== ρα_0_slice)
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
