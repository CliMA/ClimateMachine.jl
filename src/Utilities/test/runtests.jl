
using Utilities

using Test

using Utilities.MoistThermodynamics, PlanetParameters

using LinearAlgebra

@testset "moist thermodynamics" begin
  # ideal gas law and tests
  @test air_pressure([1, 1, 1], [1, 1, 2], [1, 0, 1], [0, 0, 0.5], [0, 0, 0]) ≈ [R_v, R_d, R_v]
  @test air_pressure([1, 1], [1, 2]) ≈ [R_d, 2*R_d]

  # gas constants and heat capacities
  @test gas_constant_air([0, 1, 0.5], [0, 0, 0.5], [0, 0, 0]) ≈ [R_d, R_v, R_d/2]
  @test gas_constant_air() ≈ R_d
  @test cp_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d, cp_v, cp_l, cp_i]
  @test cp_m() ≈ cp_d
  @test cv_m([0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]) ≈ [cp_d - R_d, cp_v - R_v, cv_l, cv_i]

  # specific latent heats
  @test latent_heat_vapor(T_0)  ≈ LH_v0
  @test latent_heat_fusion(T_0) ≈ LH_f0
  @test latent_heat_sublim(T_0) ≈ LH_s0

  # saturation vapor pressure and specific humidity
  p=1.e5; q_t=0.23;
  @test sat_vapor_press_liquid(T_triple) ≈ press_triple
  @test sat_vapor_press_ice(T_triple) ≈ press_triple
  @test sat_shum([T_triple, T_triple], [p, p], [0., q_t], [0., q_t/2], [0., q_t/2]) ≈
    1/molmass_ratio * press_triple / (p - press_triple) * [1, 1 - q_t]
  @test sat_shum([T_triple, T_triple], [p, p], [0., q_t]) ≈
      1/molmass_ratio * press_triple / (p - press_triple) * [1., 1-q_t]
  @test sat_shum_generic([T_triple, T_triple], [p, p], [0., q_t], phase="liquid") ≈
    1/molmass_ratio * press_triple / (p - press_triple) * [1., 1-q_t]
  @test sat_shum_generic([T_triple, T_triple], [p, p], [0., q_t], phase="ice") ≈
      1/molmass_ratio * press_triple / (p - press_triple) * [1., 1-q_t]
  @test sat_shum_generic(T_triple-20, p, q_t, phase="liquid") >=
        sat_shum_generic(T_triple-20, p, q_t, phase="ice")

  # energy functions and inverse (temperature)
  T=300; KE=11.; PE=13.;
  @test air_temperature(cv_d*(T-T_0) .* [1, 1, 1], 0, 0, 0) ≈ [T, T, T]
  @test air_temperature(cv_d*(T-T_0) .* [1, 1, 1]) ≈ [T, T, T]
  @test air_temperature(cv_m([0, q_t], 0, 0).*(T-T_0).+[0, q_t*IE_v0], [0, q_t], 0, 0) ≈ [T, T]
  @test total_energy([KE, KE, 0], [PE, PE, 0], [T_0, T, T_0], [0, 0, q_t], [0, 0, 0], [0, 0, 0]) ≈
    [KE + PE, KE + PE + cv_d*(T-T_0), q_t * IE_v0]

  # phase partitioning in equilibrium
  T   = [T_icenuc-10, T_freeze+10];
  q_l = 0.1;
  p   = [1e5, 1e5];
  q_t = [.21, .60];
  q_l_out = NaN*ones(size(T)); q_i_out = NaN*ones(size(T))
  @test liquid_fraction(T) ≈ [0, 1]
  @test liquid_fraction(T, [q_l, q_l], [q_l, q_l/2]) ≈ [0.5, 2/3]
  q_l_out, q_i_out = phase_partitioning_eq(T, p, q_t);
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
  p             = [1e5, 1e4];
  E_int         = internal_energy_sat(T_true, p, q_t);
  T, q_l, q_i   = saturation_adjustment(E_int, p, q_t, T_trial);
  @test norm(T - T_true)/length(T) < 1e-2
  @test q_t - q_l - q_i ≈ sat_shum(T, p, q_t)

  # potential temperatures
  @test liquid_ice_pottemp([T, T], [MSLP, MSLP], [0, 0], [0, 0], [0, 0]) ≈ [T, T]
  @test liquid_ice_pottemp([T, T], .1*[MSLP, MSLP], [0, 1], [0, 0], [0, 0]) ≈
    T .* 10 .^[R_d/cp_d, R_v/cp_v]
end
