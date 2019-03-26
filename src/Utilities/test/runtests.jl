
using CLIMA.Utilities, CLIMA.PlanetParameters

using Test

using CLIMA.Utilities.MoistThermodynamics
using CLIMA.Utilities.RootSolvers

using LinearAlgebra

@testset "moist thermodynamics" begin
  # ideal gas law
  @test air_pressure.([1, 1, 1], [1, 1, 2], [1, 0, 1], [0, 0, 0.5], [0, 0, 0]) ≈ [R_v, R_d, R_v]
  @test air_pressure.([1, 1], [1, 2]) ≈ [R_d, 2*R_d]
  @test air_density.([1, 1], [1, 2]) ≈ [1/R_d, 2/R_d]

  # gas constants and heat capacities
  @test gas_constant_air.([0, 1, 0.5], [0, 0, 0.5], [0, 0, 0]) ≈ [R_d, R_v, R_d/2]
  @test gas_constant_air() ≈ R_d
  @test cp_m.([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d, cp_v, cp_l, cp_i]
  @test cp_m() ≈ cp_d
  @test cv_m.([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d - R_d, cp_v - R_v, cv_l, cv_i]

  # specific latent heats
  @test latent_heat_vapor(T_0)  ≈ LH_v0
  @test latent_heat_fusion(T_0) ≈ LH_f0
  @test latent_heat_sublim(T_0) ≈ LH_s0

  # saturation vapor pressure and specific humidity
  p=1.e5; q_t=0.23; ρ=1.;
  ρ_v_triple = press_triple / R_v / T_triple;
  @test saturation_vapor_pressure(T_triple, Liquid()) ≈ press_triple
  @test saturation_vapor_pressure(T_triple, Ice()) ≈ press_triple
  @test saturation_shum.([T_triple, T_triple], [ρ, ρ], [0., q_t/2], [0., q_t/2]) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.([T_triple, T_triple], [ρ, ρ]; phase=Liquid()) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.([T_triple, T_triple], [ρ, ρ]; phase=Ice()) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.(T_triple-20, ρ; phase=Liquid()) >=
        saturation_shum_generic.(T_triple-20, ρ; phase=Ice())

  # energy functions and inverse (temperature)
  T=300; KE=11.; PE=13.;
  @test air_temperature.(cv_d*(T-T_0) .* [1, 1, 1], 0, 0, 0) ≈ [T, T, T]
  @test air_temperature.(cv_d*(T-T_0) .* [1, 1, 1]) ≈ [T, T, T]
  @test air_temperature.(cv_m.([0, q_t], 0, 0).*(T-T_0).+[0, q_t*IE_v0], [0, q_t], 0, 0) ≈ [T, T]
  @test total_energy.([KE, KE, 0], [PE, PE, 0], [T_0, T, T_0], [0, 0, q_t], [0, 0, 0], [0, 0, 0]) ≈
    [KE + PE, KE + PE + cv_d*(T-T_0), q_t * IE_v0]

  # phase partitioning in equilibrium
  T   = [T_icenuc-10, T_freeze+10];
  q_l = 0.1;
  ρ   = [1., .1];
  q_t = [.21, .60];
  q_l_out = zeros(size(T)); q_i_out = zeros(size(T))
  @test liquid_fraction.(T) ≈ [0, 1]
  @test liquid_fraction.(T, [q_l, q_l], [q_l, q_l/2]) ≈ [0.5, 2/3]
  phase_partitioning_eq!(q_l_out, q_i_out, T, ρ, q_t);
    @test q_l_out[1] ≈ 0
    @test q_l_out[2] > 0
    @test q_l_out[2] <= q_t[2]
    @test q_i_out[1] > 0
    @test q_i_out[1] <= q_t[1]
    @test q_i_out[2] ≈ 0

  # saturation adjustment in equilibrium (i.e., given the thermodynamic
  # variables E_int, p, q_t, compute the temperature and partitioning of the phases
  T_true        = [200., 300.];
  T_trial       = [220., 290.];
  q_t           = [.21, .78];
  ρ             = [.1, 1];
  E_int         = internal_energy_sat.(T_true, ρ, q_t);
  T             = saturation_adjustment.(E_int, ρ, q_t, T_trial);
  @test norm(T - T_true)/length(T) < 1e-2
  # @test all(T .≈ T_true)

  # corresponding phase partitioning
  q_l_out = zeros(size(T)); q_i_out = zeros(size(T));
  phase_partitioning_eq!(q_l_out, q_i_out, T, ρ, q_t);

  @test q_t - q_l_out - q_i_out ≈ saturation_shum.(T, ρ)

  # potential temperatures
  T = 300;
  @test liquid_ice_pottemp.([T, T], [MSLP, MSLP], [0, 0], [0, 0], [0, 0]) ≈ [T, T]
  @test liquid_ice_pottemp.([T, T], .1*[MSLP, MSLP], [0, 1], [0, 0], [0, 0]) ≈
    T .* 10 .^[R_d/cp_d, R_v/cp_v]

  # dry potential temperatures. FIXME: add correctness tests
  T = 300; p=1.e5; q_t=0.23
  @test dry_pottemp(T, p, q_t) isa typeof(p)

  # Exner function. FIXME: add correctness tests
  p=1.e5; q_t=0.23
  @test exner(p, q_t) isa typeof(p)

end

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
