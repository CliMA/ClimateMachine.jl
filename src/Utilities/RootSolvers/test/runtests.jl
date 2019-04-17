using Test
using CLIMA.RootSolvers

const HAVE_CUDA = Base.identify_package("CuArrays") !== nothing

@testset "RootSolvers correctness" begin
  f(x) = x^2 - 100^2
  for T in [Float32, Float64]
    x_root, converged = find_zero(f, T(0.0), T(1000.0), SecantMethod())
    @test converged
    @test x_root isa T
    @test x_root ≈ 100

    x_root, converged = find_zero(f, T(0.0), T(1000.0), RegulaFalsiMethod())
    @test converged
    @test x_root isa T
    @test x_root ≈ 100

    x_root, converged = find_zero(f, T(1.0), NewtonsMethod())
    @test converged
    @test x_root isa T
    @test x_root ≈ 100
  end  
end

@static if HAVE_CUDA
  using CuArrays
  CuArrays.allowscalar(false)
  
  @testset "CUDA RootSolvers" begin
    X0 = cu(rand(5,5))
    X1 = cu(rand(5,5)) .+ 1000
    f(x) = x^2 - 100^2

    result = RootSolvers.find_zero.(f, X0, X1, SecantMethod())
    X_roots = first.(result)
    converged = last.(result)
    @test all(converged)
    @test eltype(X_roots) == eltype(X0)
    @test all(X_roots .≈ 100)
  end
end
