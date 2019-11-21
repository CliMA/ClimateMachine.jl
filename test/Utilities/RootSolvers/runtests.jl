using Test
using CLIMA
using CLIMA.RootSolvers

@testset "RootSolvers - solution correctness" begin
  f(x) = x^2 - 100^2
  for T in [Float32, Float64]
    sol = find_zero(f, T(0.0), T(1000.0), SecantMethod())
    @test sol.converged
    @test sol.root isa T
    @test sol.root ≈ 100

    sol = find_zero(f, T(0.0), T(1000.0), RegulaFalsiMethod())
    @test sol.converged
    @test sol.root isa T
    @test sol.root ≈ 100

    sol = find_zero(f, T(1.0), NewtonsMethod())
    @test sol.converged
    @test sol.root isa T
    @test sol.root ≈ 100
  end
end

@static if haspkg("CuArrays")
  using CuArrays
  CuArrays.allowscalar(false)

  @testset "RootSolvers CUDA - solution " begin
    for T in [Float32, Float64]
      X0 = cu(rand(T, 5,5))
      X1 = cu(rand(T, 5,5)) .+ 1000
      f(x) = x^2 - 100^2

      sol = RootSolvers.find_zero.(f, X0, X1, SecantMethod())
      converged = map(x->x.converged, sol)
      X_roots = map(x->x.root, sol)
      @test all(converged)
      @test eltype(X_roots) == eltype(X0)
      @test all(X_roots .≈ 100)
    end
  end

end
