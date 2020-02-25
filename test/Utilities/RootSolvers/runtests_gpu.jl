using Test
using CLIMA
using CLIMA.RootSolvers

CLIMA.init()

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
