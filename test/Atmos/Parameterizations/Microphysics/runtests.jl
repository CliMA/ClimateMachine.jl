using Test
using ClimateMachine.Microphysics
using ClimateMachine.MoistThermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics: τ_cond_evap, τ_acnv, q_liq_threshold

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

@testset "RainDropFallSpeed" begin
    # two typical rain drop sizes
    r_small = 0.5 * 1e-3
    r_big = 3.5 * 1e-3

    # example atmospheric conditions
    p_range = [1013.0, 900.0, 800.0, 700.0, 600.0, 500.0] .* 100
    T_range = [20.0, 20.0, 15.0, 10.0, 0.0, -10.0] .+ 273.15
    ρ_range = p_range ./ R_d(param_set) ./ T_range

    # previousely calculated terminal velocity values
    ref_term_vel_small = [4.44, 4.71, 4.96, 5.25, 5.57, 5.99]
    ref_term_vel_big = [11.75, 12.47, 13.11, 13.90, 14.74, 15.85]

    for idx in range(Int(1), stop = Int(6))
        vc = terminal_velocity_single_drop_coeff(param_set, ρ_range[idx])

        term_vel_small = vc .* sqrt(r_small .* grav(param_set))
        term_vel_big = vc .* sqrt(r_big .* grav(param_set))

        @test term_vel_small ≈ ref_term_vel_small[idx] atol = 0.01
        @test term_vel_big ≈ ref_term_vel_big[idx] atol = 0.01
    end
end

@testset "RainFallSpeed" begin

    # eq. 5d in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function terminal_velocity_empir(
        q_rai::FT,
        q_tot::FT,
        ρ::FT,
        ρ_air_ground::FT,
    ) where {FT <: Real}
        rr = q_rai / (1 - q_tot)
        vel = FT(14.34) * ρ_air_ground^FT(0.5) * ρ^-FT(0.3654) * rr^FT(0.1346)
        return vel
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_tot, ρ_air_ground = 1.2, 20 * 1e-3, 1.22

    for q_rai in q_rain_range
        @test terminal_velocity(param_set, q_rai, ρ_air) ≈
              terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground) atol =
            0.2 * terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground)
    end
end

@testset "CloudCondEvap" begin

    q_liq_sat = 5e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    for fr in frac
        q_liq = q_liq_sat * fr
        @test conv_q_vap_to_q_liq(
            param_set,
            PhasePartition(0.0, q_liq_sat, 0.0),
            PhasePartition(0.0, q_liq, 0.0),
        ) ≈ (1 - fr) * q_liq_sat / τ_cond_evap(param_set)
    end
end

@testset "RainAutoconversion" begin

    _q_liq_threshold = q_liq_threshold(param_set)
    _τ_acnv = τ_acnv(param_set)
    q_liq_small = 0.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai_acnv(param_set, q_liq_small) == 0.0

    q_liq_big = 1.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai_acnv(param_set, q_liq_big) ==
          0.5 * _q_liq_threshold / _τ_acnv
end

@testset "RainAccretion" begin

    # eq. 5b in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function accretion_empir(q_rai::FT, q_liq::FT, q_tot::FT) where {FT <: Real}
        rr = q_rai / (FT(1) - q_tot)
        rl = q_liq / (FT(1) - q_tot)
        return FT(2.2) * rl * rr^FT(7 / 8)
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_liq, q_tot = 1.2, 5e-4, 20e-3

    for q_rai in q_rain_range
        @test conv_q_liq_to_q_rai_accr(param_set, q_liq, q_rai, ρ_air) ≈
              accretion_empir(q_rai, q_liq, q_tot) atol =
            0.1 * accretion_empir(q_rai, q_liq, q_tot)
    end
end

@testset "RainEvaporation" begin

    # eq. 5c in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function rain_evap_empir(
        param_set::AbstractParameterSet,
        q_rai::FT,
        q::PhasePartition,
        T::FT,
        p::FT,
        ρ::FT,
    ) where {FT <: Real}

        q_sat = q_vap_saturation(param_set, T, ρ, q)
        q_vap = q.tot - q.liq
        rr = q_rai / (1 - q.tot)
        rv_sat = q_sat / (1 - q.tot)
        S = q_vap / q_sat - 1

        ag, bg = FT(5.4 * 1e2), FT(2.55 * 1e5)
        G = FT(1) / (ag + bg / p / rv_sat) / ρ

        av, bv = FT(1.6), FT(124.9)
        F =
            av * (ρ / FT(1e3))^FT(0.525) * rr^FT(0.525) +
            bv * (ρ / FT(1e3))^FT(0.7296) * rr^FT(0.7296)

        return 1 / (1 - q.tot) * S * F * G
    end

    # example values
    T, p = 273.15 + 15, 90000.0
    ϵ = 1.0 / molmass_ratio(param_set)
    p_sat = saturation_vapor_pressure(param_set, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    q_tot = 15e-3
    q_vap = 0.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(param_set, q)
    ρ = p / R / T

    for q_rai in q_rain_range
        @test conv_q_rai_to_q_vap(param_set, q_rai, q, T, p, ρ) ≈
              rain_evap_empir(param_set, q_rai, q, T, p, ρ) atol =
            -0.5 * rain_evap_empir(param_set, q_rai, q, T, p, ρ)
    end
end
