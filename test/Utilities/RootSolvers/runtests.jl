using Test
using CLIMA
using CLIMA.RootSolvers

CLIMA.init()

@testset "RootSolvers - compact solution correctness" begin
  f(x) = x^2 - 100^2
  f′(x) = 2x
  for FT in [Float32, Float64]
    sol = find_zero(f, FT(0.0), FT(1000.0), SecantMethod(), CompactSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100

    sol = find_zero(f, FT(0.0), FT(1000.0), RegulaFalsiMethod(), CompactSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100

    sol = find_zero(f, FT(1.0), NewtonsMethodAD(), CompactSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100

    sol = find_zero(f, f′, FT(1.0), NewtonsMethod(), CompactSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100
  end
end

@testset "RootSolvers - verbose solution correctness" begin
  f(x) = x^2 - 100^2
  f′(x) = 2x
  for FT in [Float32, Float64]
    sol = find_zero(f, FT(0.0), FT(1000.0), SecantMethod(), VerboseSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100
    @test sol.err < 1e-3
    @test sol.iter_performed < 20
    @test sol.iter_performed+1 == length(sol.root_history) == length(sol.err_history)

    sol = find_zero(f, FT(0.0), FT(1000.0), RegulaFalsiMethod(), VerboseSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100
    @test sol.err < 1e-3
    @test sol.iter_performed < 20
    @test sol.iter_performed+1 == length(sol.root_history) == length(sol.err_history)

    sol = find_zero(f, FT(1.0), NewtonsMethodAD(), VerboseSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100
    @test sol.err < 1e-3
    @test sol.iter_performed < 20
    @test sol.iter_performed+1 == length(sol.root_history) == length(sol.err_history)

    sol = find_zero(f, f′, FT(1.0), NewtonsMethod(), VerboseSolution())
    @test sol.converged
    @test sol.root isa FT
    @test sol.root ≈ 100
    @test sol.err < 1e-3
    @test sol.iter_performed < 20
    @test sol.iter_performed+1 == length(sol.root_history) == length(sol.err_history)
  end
end

if CLIMA.array_type() != Array
  CLIMA.gpu_allowscalar(false)

  @testset "RootSolvers CUDA - compact solution " begin
    for FT in [Float32, Float64]
      X0 = cu(rand(FT, 5,5))
      X1 = cu(rand(FT, 5,5)) .+ 1000
      f(x) = x^2 - 100^2

      sol = RootSolvers.find_zero.(f, X0, X1, SecantMethod(), CompactSolution())
      converged = map(x->x.converged, sol)
      X_roots = map(x->x.root, sol)
      @test all(converged)
      @test eltype(X_roots) == eltype(X0)
      @test all(X_roots .≈ 100)
    end
  end

  @testset "RootSolvers CUDA - verbose solution " begin
    for FT in [Float32, Float64]
      X0 = cu(rand(FT, 5,5))
      X1 = cu(rand(FT, 5,5)) .+ 1000
      f(x) = x^2 - 100^2

      sol = RootSolvers.find_zero.(f, X0, X1, SecantMethod(), VerboseSolution())
      converged = map(x->x.converged, sol)
      X_roots = map(x->x.root, sol)
      err = map(x->x.err, sol)
      iter_performed = map(x->x.iter_performed, sol)
      @test all(converged)
      @test eltype(X_roots) == eltype(X0)
      @test all(X_roots .≈ 100)
      @test all(err .< 1e-2)
      @test all(iter_performed .< 20)
    end
  end
end
