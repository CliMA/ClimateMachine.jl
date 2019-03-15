using Test

using Grids
using StateVecs

N_subdomains = 3 # number of sub-domains
N_legendre_poly = 1 # number of Legendre polynomials (great candidate for SVector/overloaded +-*/)
N_elems = 10       # number of elements

grid = Grid(0.0, 1.0, N_elems, N_legendre_poly)

@testset "Simple memory access" begin
  vars = (
    (:ρ_0, 1),
    (:w, N_subdomains),
    (:a, N_subdomains),
    (:α_0, 1),
    )
  state_vec = StateVec(vars, grid, N_legendre_poly)
  state_vec[:ρ_0, 1, 1] = 2.0
  @test state_vec[:ρ_0, 1, 1] == 2.0
end

@testset "Bounds check" begin
  L = try
    vars = (
      (:ρ_0, 1),
      (:w, N_subdomains),
      (:a, N_subdomains),
      (:α_0, 1),
      )
    state_vec = StateVec(vars, grid, N_legendre_poly)
    for e in over_elems(grid)
      ρ_0_e = state_vec[:ρ_0, e]
      for i in over_sub_domains(state_vec)
        w_0_e_i = state_vec[:w, e, i]
      end
    end
    true
  catch
    false
  end
  @test L
end
