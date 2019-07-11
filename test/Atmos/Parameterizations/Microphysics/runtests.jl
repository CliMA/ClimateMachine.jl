using Test
using CLIMA.Microphysics
using CLIMA.MicrophysicsParameters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters

@testset "RainDropFallSpeed" begin
  # two typical rain drop sizes
  r_small = 0.5 * 1e-3
  r_big   = 3.5 * 1e-3

  # example atmospheric conditions
  p_range = [1013., 900., 800., 700., 600., 500.] .* 100
  T_range = [  20.,  20.,  15.,  10.,   0., -10.] .+ 273.15
  ρ_range = p_range ./ R_d ./ T_range

  # previousely calculated terminal velocity values
  ref_term_vel_small = [4.44,  4.71,  4.96, 5.25, 5.57, 5.99]
  ref_term_vel_big   = [11.75, 12.47, 13.11, 13.90, 14.74, 15.85]

  for idx in range(Int(1), stop=Int(6))
    vc = terminal_velocity_single_drop_coeff(ρ_range[idx])

    term_vel_small = vc .* sqrt(r_small .* grav)
    term_vel_big   = vc .* sqrt(r_big   .* grav)

    @test term_vel_small ≈ ref_term_vel_small[idx] atol = 0.01
    @test term_vel_big   ≈ ref_term_vel_big[idx]   atol = 0.01
  end
end

@testset "RainFallSpeed" begin

  # eq. 5d in Smolarkiewicz and Grabowski 1996
  # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
  function terminal_velocity_empir(q_rai::DT, q_tot::DT, ρ::DT,
                                      ρ_air_ground::DT) where {DT<:Real}
      rr  = q_rai / (1 - q_tot)
      vel = DT(14.34) * ρ_air_ground^DT(0.5) * ρ^-DT(0.3654) * rr^DT(0.1346)
      return vel
  end

  # some example values
  q_rain_range = range(1e-8, stop=5e-3, length=10)
  ρ_air, q_tot, ρ_air_ground = 1.2, 20 * 1e-3, 1.22

  for q_rai in q_rain_range
    @test terminal_velocity(q_rai, ρ_air) ≈
          terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground) atol =
            0.2 * terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground)
  end
end

@testset "CloudCondEvap" begin

  q_liq_sat = 5e-3
  frac = [0., 0.5, 1., 1.5]

  for fr in frac
   q_liq = q_liq_sat * fr
   @test conv_q_vap_to_q_liq(PhasePartition(0., q_liq_sat, 0.),
                             PhasePartition(0., q_liq, 0.)) ≈
         (1 - fr) * q_liq_sat / τ_cond_evap
  end
end

@testset "RainAutoconversion" begin

  q_liq_small = 0.5 * q_liq_threshold
  @test conv_q_liq_to_q_rai_acnv(q_liq_small) == 0.

  q_liq_big = 1.5 * q_liq_threshold
  @test conv_q_liq_to_q_rai_acnv(q_liq_big) ==
        0.5 * q_liq_threshold / τ_acnv
end

@testset "RainAccretion" begin

  # eq. 5b in Smolarkiewicz and Grabowski 1996
  # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
  function accretion_empir(q_rai::DT, q_liq::DT, q_tot::DT) where {DT<:Real}
      rr  = q_rai / (DT(1) - q_tot)
      rl  = q_liq / (DT(1) - q_tot)
      return DT(2.2) * rl * rr^DT(7/8)
  end

  # some example values
  q_rain_range = range(1e-8, stop=5e-3, length=10)
  ρ_air, q_liq, q_tot = 1.2, 5e-4, 20e-3

  for q_rai in q_rain_range
    @test conv_q_liq_to_q_rai_accr(q_liq, q_rai, ρ_air) ≈
          accretion_empir(q_rai, q_liq, q_tot) atol =
            0.1 * accretion_empir(q_rai, q_liq, q_tot)
  end
end

@testset "RainEvaporation" begin

  # eq. 5c in Smolarkiewicz and Grabowski 1996
  # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
  function rain_evap_empir(q_rai::DT, q::PhasePartition,
                           T::DT, p::DT, ρ::DT) where {DT<:Real}

      q_sat  = q_vap_saturation(T, ρ, q)
      q_vap  = q.tot - q.liq
      rr     = q_rai / (1 - q.tot)
      rv_sat = q_sat / (1 - q.tot)
      S      = q_vap/q_sat - 1

      ag, bg = DT(5.4 * 1e2), DT(2.55 * 1e5)
      G = DT(1) / (ag + bg / p / rv_sat) / ρ

      av, bv = DT(1.6), DT(124.9)
      F = av * (ρ/DT(1e3))^DT(0.525)  * rr^DT(0.525) +
          bv * (ρ/DT(1e3))^DT(0.7296) * rr^DT(0.7296)

      return 1 / (1 - q.tot) * S * F * G
  end

  # example values
  T, p = 273.15 + 15, 90000.
  ϵ = 1. / molmass_ratio
  p_sat = saturation_vapor_pressure(T, Liquid())
  q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.))
  q_rain_range = range(1e-8, stop=5e-3, length=10)
  q_tot = 15e-3
  q_vap = 0.15 * q_sat
  q_ice = 0.
  q_liq = q_tot - q_vap - q_ice
  q = PhasePartition(q_tot, q_liq, q_ice)
  R = gas_constant_air(q)
  ρ = p / R / T

  for q_rai in q_rain_range
    @test conv_q_rai_to_q_vap(q_rai, q, T, p, ρ) ≈
          rain_evap_empir(q_rai, q, T, p, ρ) atol =
            -0.5 * rain_evap_empir(q_rai, q, T, p, ρ)
  end
end
