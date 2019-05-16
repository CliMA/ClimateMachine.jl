using Pkg, Test

using CLIMA.TurbulenceConvection.Grids
using CLIMA.TurbulenceConvection.StateVecs
using CLIMA.TurbulenceConvection.StateVecFuncs

n_subdomains = 3 # number of sub-domains
n_elems_real = 10 # number of elements

grid = Grid(0.0, 1.0, n_elems_real)
vars = ( (:ρ_0, 1), (:a, n_subdomains), (:w, n_subdomains),
         (:ϕ, n_subdomains), (:ψ, n_subdomains) )
state_vec = StateVec(vars, grid)
vars = ((:w_ave, 1), (:cv_ϕ_ψ, n_subdomains), (:TCV_ϕ_ψ, 1))
tmp = StateVec(vars, grid)

@testset "Export fields" begin
  for k in over_elems(grid)
    state_vec[:ρ_0, k] = 1*k
    for i in over_sub_domains(state_vec)
      state_vec[:w, k, i] = 2*k
      state_vec[:a, k, i] = 3*k
    end
  end
  export_state(state_vec, "./", "state_vec")
  export_state(state_vec, grid, "./", "state_vec")
  export_state(state_vec, grid, "./", "state_vec", UseVTK())
end

@testset "Assign ghost, extrapolate, surface funcs" begin
  for k in over_elems(grid)
    state_vec[:a, k] = 2
  end

  assign_ghost!(state_vec, :a, 0.0, grid)
  @test all(state_vec[:a, k, 1] ≈ 0.0 for k in over_elems_ghost(grid))

  @test surface_val(state_vec, :a, grid) ≈ 1
  @test first_elem_above_surface_val(state_vec, :a, grid) ≈ 2

  extrap!(state_vec, :a, grid, 1)
  k = 1+grid.n_ghost
  @test state_vec[:a, k-1] ≈ 2*state_vec[:a, k] - state_vec[:a, k+1]
  k = grid.n_elem-grid.n_ghost
  @test state_vec[:a, k+1] ≈ 2*state_vec[:a, k] - state_vec[:a, k-1]
end

@testset "Domain average" begin
  state_vec[:a, 1, 1] = 0.25
  state_vec[:a, 1, 2] = 0.75
  state_vec[:a, 1, 3] = 0
  state_vec[:w, 1, 1] = 2
  state_vec[:w, 1, 2] = 2
  state_vec[:w, 1, 3] = 2
  domain_average!(tmp, state_vec, state_vec, (:w_ave,), (:w,), :a, grid)
  @test tmp[:w_ave, 1] ≈ 2
end

@testset "Distribute" begin
  tmp[:w_ave, 1] = 2

  distribute!(state_vec, tmp, (:w,), (:w_ave,), grid)
  @test state_vec[:w, 1, 1] ≈ 2
  @test state_vec[:w, 1, 2] ≈ 2
  @test state_vec[:w, 1, 3] ≈ 2
end

@testset "Total covariance" begin
  state_vec[:a, 1, 1] = 0.1
  state_vec[:a, 1, 2] = 0.2
  state_vec[:a, 1, 3] = 0.3
  state_vec[:ϕ, 1, 1] = 1
  state_vec[:ϕ, 1, 2] = 2
  state_vec[:ϕ, 1, 3] = 3
  state_vec[:ψ, 1, 1] = 2
  state_vec[:ψ, 1, 2] = 3
  state_vec[:ψ, 1, 3] = 4
  tmp[:cv_ϕ_ψ, 1, 1] = 1.0
  tmp[:cv_ϕ_ψ, 1, 2] = 1.0
  tmp[:cv_ϕ_ψ, 1, 3] = 1.0
  decompose_ϕ_ψ(cv) = cv == :cv_ϕ_ψ ?  (:ϕ , :ψ) : error("Bad init")
  total_covariance!(tmp, state_vec, tmp, state_vec,
                    (:TCV_ϕ_ψ,), (:cv_ϕ_ψ,), :a, grid, decompose_ϕ_ψ)
  @test tmp[:TCV_ϕ_ψ, 1] ≈ 0.8
end

@testset "Integrate ODE" begin
  c_0 = 3.0
  y_0 = 2.0
  func(z, args) = y_0 + z
  sol_correct = c_0 .+ [y_0*get_z(grid, k) + 1/2*get_z(grid, k)^2 for k in over_elems_real(grid)]
  integrate_ode!(state_vec, :w, grid, func, c_0, NamedTuple(), 1)
  sol_approx = [state_vec[:w, k] for k in over_elems_real(grid)]
  sol_percent_error = [(sol_approx[k]-sol_correct[k])/sol_correct[k]*100 for k in 1:length(sol_correct)]
  @test all(sol_percent_error .< 100*grid.dz)
end

