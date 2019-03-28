using Test
using CLIMA.RootSolvers

@testset "RootSolvers correctness" begin
  for m in RootSolvers.get_solve_methods()
    x_star2 = 10000.0
    f(x, y) = x^2 - y
    x_star = sqrt(x_star2)
    x_0 = 0.0
    x_1 = 1.0
    args = Tuple(x_star2)
    tol_abs = 1.0e-3
    iter_max = 100
    x_root, converged = RootSolvers.find_zero(f,
                                              x_0, x_1,
                                              args,
                                              IterParams(tol_abs, iter_max),
                                              SecantMethod()
                                              )
    @test abs(x_root - x_star) < tol_abs
  end
end

@testset "RootSolvers convergence" begin
  for m in RootSolvers.get_solve_methods()
    for i in 1:4
      f(x) = sum([rand(1)[1]-0.5 + rand(1)[1]*x^j for j in 1:i])
      x_0 = rand(1)[1]+0.0
      x_1 = rand(1)[1]+1.0
      args = ()
      tol_abs = 1.0e-3
      iter_max = 10000
      x_root, converged = RootSolvers.find_zero(f,
                                                x_0, x_1,
                                                args,
                                                IterParams(tol_abs, iter_max),
                                                m
                                                )
      @test converged
    end
  end
end

const HAVE_CUDA = try
  using CUDAdrv
  using CUDAnative
  using CuArrays
  true
catch
  false
end
@testset "CUDA RootSolvers" begin
  if HAVE_CUDA
    for m in [RootSolvers.SecantMethod()]             
      x_ca = cu(rand(5, 5))
      x_ca_0 = x_ca
      x_ca_1 = x_ca.+2
      
      t = typeof(x_ca[1])
      x_star2 = t(10000.0)
      f(x) = x^2 - x_star2
      f(x, y) = x^2 - y
      args = Tuple(x_star2)
      x_star = sqrt(x_star2)
      x_0 = t(0.0)
      x_1 = t(1.0)
      tol_abs = t(1.0e-3)
      iter_max = 100

      # Test args method
      x_root, converged = find_zero(f, x_0, x_1, args, IterParams(tol_abs, iter_max), m)
      @test x_root ≈ 100.0
      R = find_zero.(f, x_ca_0, x_ca_1, Ref(args), Ref(IterParams(tol_abs, iter_max)), Ref(m))
      x_roots = [x for (x, y) in R]
      @test all(x_roots .≈ 100.0)
    end
  end
end
