using Test
using CLIMA.MoistThermodynamics

using CLIMA.PlanetParameters
using LinearAlgebra

@testset "moist thermodynamics" begin
  DT = Float64
  # ideal gas law
  @test air_pressure(DT(1), DT(1), PhasePartition(DT(1))) === DT(R_v)
  @test air_pressure(DT(1), DT(2), PhasePartition(DT(1), DT(0.5), DT(0))) === DT(R_v)
  @test air_pressure(DT(1), DT(1)) === DT(R_d)
  @test air_pressure(DT(1), DT(2)) === 2*DT(R_d)
  @test air_density(DT(1), DT(1)) === 1/DT(R_d)
  @test air_density(DT(1), DT(2)) === 2/DT(R_d)

  # gas constants and heat capacities
  @test gas_constant_air(PhasePartition(DT(0))) === DT(R_d)
  @test gas_constant_air(PhasePartition(DT(1))) === DT(R_v)
  @test gas_constant_air(PhasePartition(DT(0.5), DT(0.5))) ≈ DT(R_d)/2
  @test gas_constant_air() == R_d

  @test cp_m(PhasePartition(DT(0))) === DT(cp_d)
  @test cp_m(PhasePartition(DT(1))) === DT(cp_v)
  @test cp_m(PhasePartition(DT(1), DT(1))) === DT(cp_l)
  @test cp_m(PhasePartition(DT(1), DT(0), DT(1))) === DT(cp_i)
  @test cp_m() == cp_d

  @test cv_m(PhasePartition(DT(0))) === DT(cp_d - R_d)
  @test cv_m(PhasePartition(DT(1))) === DT(cp_v - R_v)
  @test cv_m(PhasePartition(DT(1), DT(1))) === DT(cv_l)
  @test cv_m(PhasePartition(DT(1), DT(0), DT(1))) === DT(cv_i)

  # speed of sound
  @test soundspeed_air(T_0 + 20, PhasePartition(DT(0))) == sqrt(cp_d/cv_d * R_d * (T_0 + 20))
  @test soundspeed_air(T_0 + 100, PhasePartition(DT(1))) == sqrt(cp_v/cv_v * R_v * (T_0 + 100))

  # specific latent heats
  @test latent_heat_vapor(DT(T_0))  ≈ LH_v0
  @test latent_heat_fusion(DT(T_0)) ≈ LH_f0
  @test latent_heat_sublim(DT(T_0)) ≈ LH_s0

  # saturation vapor pressure and specific humidity
  p=DT(1.e5); q_tot=DT(0.23); ρ=DT(1.);
  ρ_v_triple = press_triple / R_v / T_triple;
  @test saturation_vapor_pressure(DT(T_triple), Liquid()) ≈ press_triple
  @test saturation_vapor_pressure(DT(T_triple), Ice()) ≈ press_triple

  @test q_vap_saturation(DT(T_triple), ρ, PhasePartition(DT(0))) == ρ_v_triple / ρ
  @test q_vap_saturation(DT(T_triple), ρ, PhasePartition(q_tot,q_tot)) == ρ_v_triple / ρ

  @test q_vap_saturation_generic(DT(T_triple), ρ; phase=Liquid()) == ρ_v_triple / ρ
  @test q_vap_saturation_generic(DT(T_triple), ρ; phase=Ice()) == ρ_v_triple / ρ
  @test q_vap_saturation_generic.(DT(T_triple-20), ρ; phase=Liquid()) >=
        q_vap_saturation_generic.(DT(T_triple-20), ρ; phase=Ice())

  @test saturation_excess(DT(T_triple), ρ, PhasePartition(q_tot)) ==  q_tot - ρ_v_triple/ρ
  @test saturation_excess(DT(T_triple), ρ, PhasePartition(q_tot/1000)) == 0.0

  # energy functions and inverse (temperature)
  T=DT(300); e_kin=DT(11); e_pot=DT(13);
  @test air_temperature(DT(cv_d*(T-T_0))) === DT(T)
  @test air_temperature(DT(cv_d*(T-T_0)), PhasePartition(DT(0))) === DT(T)

  @test air_temperature(cv_m(PhasePartition(DT(0)))*(T-T_0), PhasePartition(DT(0))) === DT(T)
  @test air_temperature(cv_m(PhasePartition(DT(q_tot)))*(T-T_0) + q_tot*e_int_v0, PhasePartition(q_tot)) ≈ DT(T)

  @test total_energy(DT(e_kin),DT(e_pot), DT(T_0)) === DT(e_kin) + DT(e_pot)
  @test total_energy(DT(e_kin),DT(e_pot), DT(T)) ≈ DT(e_kin) + DT(e_pot) + cv_d*(T-T_0)
  @test total_energy(DT(0),DT(0), DT(T_0), PhasePartition(q_tot)) ≈ q_tot*e_int_v0

  # phase partitioning in equilibrium
  q_liq = DT(0.1);
  T = DT(T_icenuc-10); ρ = DT(1.0); q_tot = DT(0.21)
  @test liquid_fraction_equil(T) === DT(0)
  @test liquid_fraction_equil(T, PhasePartition(q_tot,q_liq,q_liq)) === DT(0.5)
  q = PhasePartition_equil(T, ρ, q_tot)
  @test q.liq ≈ DT(0)
  @test 0 < q.ice <= q_tot

  T = DT(T_freeze+10); ρ = DT(0.1); q_tot = DT(0.60)
  @test liquid_fraction_equil(T) === DT(1)
  @test liquid_fraction_equil(T, PhasePartition(q_tot,q_liq,q_liq/2)) === DT(2/3)
  q = PhasePartition_equil(T, ρ, q_tot)
  @test 0 < q.liq <= q_tot
  @test q.ice ≈ 0

  # saturation adjustment in equilibrium (i.e., given the thermodynamic
  # variables E_int, p, q_tot, compute the temperature and partitioning of the phases
  q_tot = DT(0); ρ = DT(1)
  @test saturation_adjustment(internal_energy_sat(300.0, ρ, q_tot), ρ, q_tot) ≈ 300.0

  q_tot = DT(0.21); ρ = DT(0.1)
  @test saturation_adjustment(internal_energy_sat(200.0, ρ, q_tot), ρ, q_tot) ≈ 200.0
  q = PhasePartition_equil(T, ρ, q_tot)
  @test q.tot - q.liq - q.ice ≈ q_vap_saturation(T, ρ)

  q_tot = DT(0.78); ρ = DT(1)
  @test saturation_adjustment(internal_energy_sat(300.0, ρ, q_tot), ρ, q_tot) ≈ 300.0
  q = PhasePartition_equil(T, ρ, q_tot)
  @test q.tot - q.liq - q.ice ≈ q_vap_saturation(T, ρ)

  # potential temperatures
  T = DT(300)
  @test liquid_ice_pottemp(T, DT(MSLP)) === T
  @test liquid_ice_pottemp(T, DT(MSLP)/10) ≈ T * 10^(R_d/cp_d)
  @test liquid_ice_pottemp(T, DT(MSLP)/10, PhasePartition(DT(1))) ≈ T * 10^(R_v/cp_v)

  # dry potential temperatures. FIXME: add correctness tests
  T = DT(300); p=DT(1.e5); q_tot=DT(0.23)
  @test dry_pottemp(T, p, PhasePartition(q_tot)) isa typeof(p)
  @test air_temperature_from_liquid_ice_pottemp(dry_pottemp(T, p, PhasePartition(q_tot)), p, PhasePartition(q_tot)) ≈ T

  # Exner function. FIXME: add correctness tests
  p=DT(1.e5); q_tot=DT(0.23)
  @test exner(p, PhasePartition(q_tot)) isa typeof(p)

  DT = Float32
  ρ = DT(1.0)
  p = DT(1000.0*100)
  e_int = DT(2.0)
  q_tot = DT(0.01)
  q_ice = DT(0.001)
  q_liq = DT(0.001)
  θ_liq_ice = DT(300.0)
  ts_eq = PhaseEquil(e_int, q_tot, ρ)
  ts_neq = PhaseNonEquil(e_int, PhasePartition(q_tot, q_liq, q_ice), ρ)
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
    @test q_vap_saturation(ts) isa typeof(e_int)
    @test saturation_excess(ts) isa typeof(e_int)
    @test liquid_fraction_equil(ts) isa typeof(e_int)
    @test liquid_fraction_nonequil(ts) isa typeof(e_int)
    @test PhasePartition(ts) isa PhasePartition{typeof(e_int)}
    @test liquid_ice_pottemp(ts) isa typeof(e_int)
    @test dry_pottemp(ts) isa typeof(e_int)
    @test exner(ts) isa typeof(e_int)
    @test liquid_ice_pottemp_sat(ts) isa typeof(e_int)
    @test specific_volume(ts) isa typeof(e_int)
    @test virtual_pottemp(ts) isa typeof(e_int)
  end

end
