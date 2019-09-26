using Pkg, Test

using CLIMA.TurbulenceConvection.FiniteDifferenceGrids
using CLIMA.TurbulenceConvection.StateVecs

output_root = joinpath("output", "tests", "StateVecFuncs")

n_subdomains = 3 # number of sub-domains
n_elems_real = 10 # number of elements

dd = DomainDecomp(gm=1,en=1,ud=1)
grid = Grid(0.0, 1.0, n_elems_real)
vars = ( (:ρ_0, DomainSubSet(gm=true)),
         (:a,   DomainSubSet(gm=true,en=true,ud=true)),
         (:w,   DomainSubSet(gm=true,en=true,ud=true)),
         (:ϕ,   DomainSubSet(gm=true,en=true,ud=true)),
         (:ψ,   DomainSubSet(gm=true,en=true,ud=true))
         )
state_vec = StateVec(vars, grid, dd)
vars = ((:cv_ϕ_ψ, DomainSubSet(gm=true,en=true,ud=true)),
        (:TCV_ϕ_ψ, DomainSubSet(gm=true)))
tmp = StateVec(vars, grid, dd)

idx = DomainIdx(state_vec)
i_gm, i_en, i_ud, i_sd, i_al = allcombinations(idx)

@testset "Export fields" begin
  for k in over_elems(grid)
    state_vec[:ρ_0, k] = 1.0*k
    for i in i_al
      state_vec[:w, k, i] = 2.0*k
      state_vec[:a, k, i] = 3.0*k
    end
  end
  export_state(state_vec, grid, output_root, "state_vec.csv")
end

@testset "Helper funcs" begin
  @test nice_string(:θ) == "theta"
  @test nice_string(:ρ) == "rho"
  @test nice_string(:∇) == "grad"
  @test nice_string(:εδ) == "entr-detr"
end

@testset "Assign ghost, extrapolate, surface funcs" begin
  for k in over_elems(grid)
    state_vec[:a, k] = 2
  end

  assign_ghost!(state_vec, :a, grid, 0.0)
  @test all(state_vec[:a, k] ≈ 0.0 for k in over_elems_ghost(grid))

  extrap!(state_vec, :a, grid)
  k = 1+grid.n_ghost
  @test state_vec[:a, k-1] ≈ 2*state_vec[:a, k] - state_vec[:a, k+1]
  k = grid.n_elem-grid.n_ghost
  @test state_vec[:a, k+1] ≈ 2*state_vec[:a, k] - state_vec[:a, k-1]
end

@testset "Domain average" begin
  state_vec[:a, 1, i_en] = 0.25
  state_vec[:a, 1, i_ud[1]] = 0.75
  state_vec[:w, 1, i_en] = 2
  state_vec[:w, 1, i_ud[1]] = 2
  domain_average!(state_vec, state_vec, :w, :a, grid)
  @test state_vec[:w, 1, i_gm] ≈ 2
end

@testset "Distribute" begin
  state_vec[:w, 1, i_gm] = 2

  distribute!(state_vec, grid, :w)
  @test state_vec[:w, 1, 1] ≈ 2
  @test state_vec[:w, 1, 2] ≈ 2
  @test state_vec[:w, 1, 3] ≈ 2
end

@testset "Total covariance" begin
  state_vec[:a, 1, 1] = 0.1
  state_vec[:a, 1, 2] = 0.2
  state_vec[:ϕ, 1, 1] = 1
  state_vec[:ϕ, 1, 2] = 2
  state_vec[:ψ, 1, 1] = 2
  state_vec[:ψ, 1, 2] = 3
  tmp[:cv_ϕ_ψ, 1, 1] = 1.0
  tmp[:cv_ϕ_ψ, 1, 2] = 1.0
  decompose_ϕ_ψ(tcv) = tcv == :TCV_ϕ_ψ ?  (:ϕ , :ψ) : error("Bad init")
  total_covariance!(tmp, state_vec, tmp, :TCV_ϕ_ψ, :cv_ϕ_ψ, :a, grid, decompose_ϕ_ψ)
  @test tmp[:TCV_ϕ_ψ, 1] ≈ 0.32
end

