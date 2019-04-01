using Test
using CLIMA.MoistThermodynamics

using CLIMA.PlanetParameters
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

  # speed of sound
  T = [T_0 + 20, T_0 + 100]; q_t = [0, 1];
  @test soundspeed_air.(T, q_t) ≈ [ sqrt(cp_d/cv_d * R_d * T[1]), sqrt(cp_v/cv_v * R_v * T[2]) ]

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
  @test saturation_excess.([T_triple, T_triple], [ρ, ρ], [q_t, q_t/1000]) ≈
        max.(0., [q_t, q_t/1000] .- ρ_v_triple / ρ * [1, 1])

  # energy functions and inverse (temperature)
  T=300; KE=11.; PE=13.;
  @test air_temperature.(cv_d*(T-T_0) .* [1, 1, 1], 0, 0, 0) ≈ [T, T, T]
  @test air_temperature.(cv_d*(T-T_0) .* [1, 1, 1]) ≈ [T, T, T]
  @test air_temperature.(cv_m.([0, q_t], 0, 0).*(T-T_0).+[0, q_t*e_int_v0], [0, q_t], 0, 0) ≈ [T, T]
  @test total_energy.([KE, KE, 0], [PE, PE, 0], [T_0, T, T_0], [0, 0, q_t], [0, 0, 0], [0, 0, 0]) ≈
    [KE + PE, KE + PE + cv_d*(T-T_0), q_t * e_int_v0]

  # phase partitioning in equilibrium
  T   = [T_icenuc-10, T_freeze+10];
  q_l = 0.1;
  ρ   = [1., .1];
  q_t = [.21, .60];
  q_l_out = zeros(size(T)); q_i_out = zeros(size(T))
  @test liquid_fraction.(T) ≈ [0, 1]
  @test liquid_fraction.(T, [q_l, q_l], [q_l, q_l/2]) ≈ [0.5, 2/3]
  result = phase_partitioning_eq.(T, ρ, q_t)
  q_l_out, q_i_out = first.(result), last.(result)
    @test q_l_out[1] ≈ 0
    @test q_l_out[2] > 0
    @test q_l_out[2] <= q_t[2]
    @test q_i_out[1] > 0
    @test q_i_out[1] <= q_t[1]
    @test q_i_out[2] ≈ 0

  # saturation adjustment in equilibrium (i.e., given the thermodynamic
  # variables e_int, p, q_t, compute the temperature and partitioning of the phases
  T_true        = [200., 300.];
  T_trial       = [220., 290.];
  q_t           = [.21, .78];
  ρ             = [.1, 1];
  e_int         = internal_energy_sat.(T_true, ρ, q_t);
  T             = saturation_adjustment.(e_int, ρ, q_t, T_trial);
  @test norm(T - T_true)/length(T) < 1e-2

  ts = InternalEnergy_Shum_eq.(e_int, q_t, ρ)
  tol = 1e-2
  temp = abs.(air_temperature.(ts) - T)
  @test all([x < tol for x in temp])

  # corresponding phase partitioning
  q_l_out = zeros(size(T)); q_i_out = zeros(size(T));
  result = phase_partitioning_eq.(T, ρ, q_t)
  q_l_out, q_i_out = first.(result), last.(result)

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
