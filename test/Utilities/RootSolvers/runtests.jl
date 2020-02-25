using Test
using CLIMA
using CLIMA.RootSolvers

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
