using Test
using CLIMA.MoistThermodynamics

using CLIMA.PlanetParameters
using LinearAlgebra

@testset "moist thermodynamics" begin
  DT = Float64
  # ideal gas law
  @test air_pressure.(DT.([1, 1, 1]), DT.([1, 1, 2]), DT.([1, 0, 1]), DT.([0, 0, 0.5]), DT.([0, 0, 0])) ≈ [R_v, R_d, R_v]
  @test air_pressure.(DT.([1, 1]), DT.([1, 2])) ≈ [R_d, 2*R_d]
  @test air_density.(DT.([1, 1]), DT.([1, 2])) ≈ [1/R_d, 2/R_d]

  # gas constants and heat capacities
  @test gas_constant_air.(DT.([0, 1, 0.5]), DT.([0, 0, 0.5]), DT.([0, 0, 0])) ≈ [R_d, R_v, R_d/2]
  @test gas_constant_air() ≈ R_d
  @test cp_m.(DT.([0, 1, 1, 1]), DT.([0, 0, 1, 0]), DT.([0, 0, 0, 1])) ≈ DT.([cp_d, cp_v, cp_l, cp_i])
  @test cp_m() ≈ cp_d
  @test cv_m.(DT.([0, 1, 1, 1]), DT.([0, 0, 1, 0]), DT.([0, 0, 0, 1])) ≈ DT.([cp_d - R_d, cp_v - R_v, cv_l, cv_i])

  # speed of sound
  T = [T_0 + 20, T_0 + 100]; q_tot = [0.0, 1.0];
  @test soundspeed_air.(T, q_tot) ≈ [ sqrt(cp_d/cv_d * R_d * T[1]), sqrt(cp_v/cv_v * R_v * T[2]) ]

  # specific latent heats
  @test latent_heat_vapor(DT(T_0))  ≈ LH_v0
  @test latent_heat_fusion(DT(T_0)) ≈ LH_f0
  @test latent_heat_sublim(DT(T_0)) ≈ LH_s0

  # saturation vapor pressure and specific humidity
  p=DT(1.e5); q_tot=DT(0.23); ρ=DT(1.);
  ρ_v_triple = press_triple / R_v / T_triple;
  @test saturation_vapor_pressure(DT(T_triple), Liquid()) ≈ press_triple
  @test saturation_vapor_pressure(DT(T_triple), Ice()) ≈ press_triple
  @test saturation_shum.(DT.([T_triple, T_triple]), [ρ, ρ], DT.([0, q_tot/2]), DT.([0, q_tot/2])) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.(DT.([T_triple, T_triple]), [ρ, ρ]; phase=Liquid()) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.(DT.([T_triple, T_triple]), [ρ, ρ]; phase=Ice()) ≈
     ρ_v_triple / ρ * [1, 1]
  @test saturation_shum_generic.(DT(T_triple-20), ρ; phase=Liquid()) >=
        saturation_shum_generic.(DT(T_triple-20), ρ; phase=Ice())
  @test saturation_excess.(DT.([T_triple, T_triple]), [ρ, ρ], DT.([q_tot, q_tot/1000])) ≈
        max.(0., [q_tot, q_tot/1000] .- ρ_v_triple / ρ * [1, 1])

  # energy functions and inverse (temperature)
  T=DT(300); e_kin=DT(11); e_pot=DT(13);
  @test air_temperature.(DT.(cv_d*(T-T_0) .* [1, 1, 1]), DT(0), DT(0), DT(0)) ≈ [T, T, T]
  @test air_temperature.(DT.(cv_d*(T-T_0) .* [1, 1, 1])) ≈ [T, T, T]
  @test air_temperature.(DT.(cv_m.([0, q_tot], DT(0), DT(0)).*(T-T_0).+[0, q_tot*e_int_v0]), DT.([0, q_tot]), DT(0), DT(0)) ≈ [T, T]
  @test total_energy.(DT.([e_kin, e_kin, 0]), DT.([e_pot, e_pot, 0]), DT.([T_0, T, T_0]), DT.([0, 0, q_tot]), DT.([0, 0, 0]), DT.([0, 0, 0])) ≈
    [e_kin + e_pot, e_kin + e_pot + cv_d*(T-T_0), q_tot * e_int_v0]

  # phase partitioning in equilibrium
  T   = DT.([T_icenuc-10, T_freeze+10];)
  q_liq = DT(0.1);
  ρ   = DT.([1., .1]);
  q_tot = DT.([.21, .60]);
  @test liquid_fraction_equil.(T) ≈ [0, 1]
  @test liquid_fraction_equil.(T, [q_liq, q_liq], [q_liq, q_liq/2]) ≈ [0.5, 2/3]
    q_result = phase_partitioning_eq.(T, ρ, q_tot);
    q_l_out = first.(q_result)
    q_i_out = last.(q_result)
    @test q_l_out[1] ≈ 0
    @test q_l_out[2] > 0
    @test q_l_out[2] <= q_tot[2]
    @test q_i_out[1] > 0
    @test q_i_out[1] <= q_tot[1]
    @test q_i_out[2] ≈ 0

  # saturation adjustment in equilibrium (i.e., given the thermodynamic
  # variables E_int, p, q_tot, compute the temperature and partitioning of the phases
  T_true        = DT.([300., 200., 300.]);
  T_trial       = DT.([T_triple, 220., 290.]);
  q_tot         = DT.([0, .21, .78]);
  ρ             = DT.([1, .1, 1]);
  E_int         = internal_energy_sat.(T_true, ρ, q_tot);
  T             = saturation_adjustment.(E_int, ρ, q_tot);
  @test norm(T - T_true)/length(T) < 1e-2

  # corresponding phase partitioning
  T_true        = DT.([200., 300.]);
  T_trial       = DT.([220., 290.]);
  q_tot         = DT.([0.21, 0.78]);
  ρ             = DT.([.1, 1]);
  E_int         = internal_energy_sat.(T_true, ρ, q_tot);
  T             = saturation_adjustment.(E_int, ρ, q_tot);
  q_result      = phase_partitioning_eq.(T, ρ, q_tot);
  q_l_out = first.(q_result)
  q_i_out = last.(q_result)
  @test q_tot - q_l_out - q_i_out ≈ saturation_shum.(T, ρ)

  # potential temperatures
  T = DT(300);
  @test liquid_ice_pottemp.([T, T], DT.(   [MSLP, MSLP]), DT.([0, 0]), DT.([0, 0]), DT.([0, 0])) ≈ [T, T]
  @test liquid_ice_pottemp.([T, T], DT.(.1*[MSLP, MSLP]), DT.([0, 1]), DT.([0, 0]), DT.([0, 0])) ≈
    T .* 10 .^[R_d/cp_d, R_v/cp_v]

  # dry potential temperatures. FIXME: add correctness tests
  T = DT(300); p=DT(1.e5); q_tot=DT(0.23)
  @test dry_pottemp(T, p, q_tot) isa typeof(p)
  @test air_temperature_from_liquid_ice_pottemp(dry_pottemp(T, p, q_tot), p, q_tot) ≈ T

  # Exner function. FIXME: add correctness tests
  p=DT(1.e5); q_tot=DT(0.23)
  @test exner(p, q_tot) isa typeof(p)

  DT = Float32
  ρ = DT(1.0)
  p = DT(1000.0*100)
  e_int = DT(2.0)
  q_tot = DT(0.01)
  q_ice = DT(0.001)
  q_liq = DT(0.001)
  θ_liq_ice = DT(300.0)
  ts_eq = PhaseEquil(e_int, q_tot, ρ)
  ts_neq = PhaseNonEquil(e_int, q_tot, q_liq, q_ice, ρ)
  ts_θ_liq_ice_eq = LiquidIcePotTempSHumEquil(θ_liq_ice, q_tot, ρ, p)
  for ts in (ts_eq, ts_neq, ts_θ_liq_ice_eq)
    @test soundspeed_air(ts) isa typeof(e_int)
    @test gas_constant_air(ts) isa typeof(e_int)
    @test air_pressure(ts) isa typeof(e_int)
    @test air_density(ts) isa typeof(e_int)
    @test cp_m(ts) isa typeof(e_int)
    @test cv_m(ts) isa typeof(e_int)
    @test eltype(moist_gas_constants(ts)) == typeof(e_int)
    @test air_temperature(ts) isa typeof(e_int)
    @test internal_energy_sat(ts) isa typeof(e_int)
    @test latent_heat_vapor(ts) isa typeof(e_int)
    @test latent_heat_sublim(ts) isa typeof(e_int)
    @test latent_heat_fusion(ts) isa typeof(e_int)
    @test saturation_shum(ts) isa typeof(e_int)
    @test saturation_excess(ts) isa typeof(e_int)
    @test liquid_fraction_equil(ts) isa typeof(e_int)
    @test liquid_fraction_nonequil(ts) isa typeof(e_int)
    @test eltype(phase_partitioning_eq(ts)) == typeof(e_int)
    @test liquid_ice_pottemp(ts) isa typeof(e_int)
    @test dry_pottemp(ts) isa typeof(e_int)
    @test exner(ts) isa typeof(e_int)
    @test liquid_ice_pottemp_sat(ts) isa typeof(e_int)
    @test specific_volume(ts) isa typeof(e_int)
    @test virtual_pottemp(ts) isa typeof(e_int)
  end

end
