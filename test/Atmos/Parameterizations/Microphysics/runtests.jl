using Test
using ClimateMachine.Microphysics
using ClimateMachine.Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end

struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liquid::L
    ice::I
    rain::R
    snow::S
end

struct EarthParamSet{M} <: AbstractEarthParameterSet
    microphys_param_set::M
end

microphys_param_set = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)

prs = EarthParamSet(microphys_param_set)
liq_prs = prs.microphys_param_set.liquid
ice_prs = prs.microphys_param_set.ice
rai_prs = prs.microphys_param_set.rain
sno_prs = prs.microphys_param_set.snow

@testset "τ_relax" begin

    @test τ_relax(liq_prs) ≈ 10
    @test τ_relax(ice_prs) ≈ 10

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
        @test terminal_velocity(prs, rai_prs, ρ_air, q_rai) ≈
              terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground) atol =
            0.2 * terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground)

    end
end

@testset "CloudLiquidCondEvap" begin

    q_liq_sat = 5e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    _τ_cond_evap = τ_relax(liq_prs)

    for fr in frac
        q_liq = q_liq_sat * fr

        @test conv_q_vap_to_q_liq_ice(
            liq_prs,
            PhasePartition(0.0, q_liq_sat, 0.0),
            PhasePartition(0.0, q_liq, 0.0),
        ) ≈ (1 - fr) * q_liq_sat / _τ_cond_evap
    end
end

@testset "CloudIceCondEvap" begin

    q_ice_sat = 2e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    _τ_cond_evap = τ_relax(ice_prs)

    for fr in frac
        q_ice = q_ice_sat * fr

        @test conv_q_vap_to_q_liq_ice(
            ice_prs,
            PhasePartition(0.0, 0.0, q_ice_sat),
            PhasePartition(0.0, 0.0, q_ice),
        ) ≈ (1 - fr) * q_ice_sat / _τ_cond_evap
    end
end

@testset "RainAutoconversion" begin

    _q_liq_threshold = q_liq_threshold(rai_prs)
    _τ_acnv = τ_acnv(rai_prs)

    q_liq_small = 0.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rai_prs, q_liq_small) == 0.0

    q_liq_big = 1.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rai_prs, q_liq_big) ==
          0.5 * _q_liq_threshold / _τ_acnv
end

@testset "SnowAutoconversion" begin

    ρ = 1.0

    # above freezing temperatures -> no snow
    q = PhasePartition(15e-3, 2e-3, 1e-3)
    T = 273.15 + 30
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # no ice -> no snow
    q = PhasePartition(15e-3, 2e-3, 0.0)
    T = 273.15 - 30
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # no supersaturation -> no snow
    T = 273.15 - 5
    q_sat_ice = q_vap_saturation_generic(prs, T, ρ; phase = Ice())
    q = PhasePartition(q_sat_ice, 2e-3, 3e-3)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same
    T = 273.15 - 10
    q_vap = 1.02 * q_vap_saturation_generic(prs, T, ρ; phase = Ice())
    q_liq = 0.0
    q_ice = 0.03 * q_vap
    q = PhasePartition(q_vap + q_liq + q_ice, q_liq, q_ice)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) ≈ 1.8512022335645584e-9

end

@testset "RainLiquidAccretion" begin

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
        @test accretion(prs, liq_prs, rai_prs, q_liq, q_rai, ρ_air) ≈
              accretion_empir(q_rai, q_liq, q_tot) atol =
            (0.1 * accretion_empir(q_rai, q_liq, q_tot))
    end
end

@testset "Accretion" begin
    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same

    # some example values
    ρ = 1.2
    q_tot = 20e-3
    q_ice = 5e-4
    q_sno = 5e-4
    q_liq = 5e-4
    q_rai = 5e-4

    @test accretion(prs, liq_prs, rai_prs, q_liq, q_rai, ρ) ≈
          1.4150106417043544e-6
    @test accretion(prs, ice_prs, sno_prs, q_ice, q_sno, ρ) ≈
          2.453070979562392e-7
    @test accretion(prs, liq_prs, sno_prs, q_liq, q_sno, ρ) ≈
          2.453070979562392e-7
    @test accretion(prs, ice_prs, rai_prs, q_ice, q_rai, ρ) ≈
          1.768763302130443e-6

    @test accretion_rain_sink(prs, ice_prs, rai_prs, q_ice, q_rai, ρ) ≈
          3.085229094251214e-5

    @test accretion_snow_rain(prs, sno_prs, rai_prs, q_sno, q_rai, ρ) ≈
          2.1705865794293408e-4
    @test accretion_snow_rain(prs, rai_prs, sno_prs, q_rai, q_sno, ρ) ≈
          6.0118801860768854e-5
end

@testset "RainEvaporation" begin

    # eq. 5c in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function rain_evap_empir(
        prs::AbstractParameterSet,
        q_rai::FT,
        q::PhasePartition,
        T::FT,
        p::FT,
        ρ::FT,
    ) where {FT <: Real}

        q_sat = q_vap_saturation_generic(prs, T, ρ; phase = Liquid())
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
    ϵ = 1.0 / molmass_ratio(prs)
    p_sat = saturation_vapor_pressure(prs, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    q_tot = 15e-3
    q_vap = 0.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(prs, q)
    ρ = p / R / T

    for q_rai in q_rain_range
        @test evaporation_sublimation(prs, rai_prs, q, q_rai, ρ, T) ≈
              rain_evap_empir(prs, q_rai, q, T, p, ρ) atol =
            -0.5 * rain_evap_empir(prs, q_rai, q, T, p, ρ)
    end

    # no condensational growth for rain
    T, p = 273.15 + 15, 90000.0
    ϵ = 1.0 / molmass_ratio(prs)
    p_sat = saturation_vapor_pressure(prs, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rai = 1e-4
    q_tot = 15e-3
    q_vap = 1.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(prs, q)
    ρ = p / R / T

    @test evaporation_sublimation(prs, rai_prs, q, q_rai, ρ, T) ≈ 0.0

end

@testset "SnowSublimation" begin
    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same

    cnt = 0
    ref_val = [
        -1.9756907119482267e-7,
        1.9751292385808357e-7,
        -1.6641552112891826e-7,
        1.663814937710236e-7,
    ]
    # some example values
    for T in [273.15 + 2, 273.15 - 2]
        p = 90000.0
        ϵ = 1.0 / molmass_ratio(prs)
        p_sat = saturation_vapor_pressure(prs, T, Ice())
        q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))

        for eps in [0.95, 1.05]
            cnt += 1

            q_tot = eps * q_sat
            q_ice = 0.0
            q_liq = 0.0
            q = PhasePartition(q_tot, q_liq, q_ice)

            q_sno = 1e-4

            R = gas_constant_air(prs, q)
            ρ = p / R / T

            @test evaporation_sublimation(prs, sno_prs, q, q_sno, ρ, T) ≈
                  ref_val[cnt]

        end
    end
end

@testset "SnowMelt" begin

    # TODO - find a good reference to compare with
    T = 273.15 + 2
    ρ = 1.2
    q_sno = 1e-4
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 9.518235437405256e-6

    # no snow -> no snow melt
    T = 273.15 + 2
    ρ = 1.2
    q_sno = 0.0
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 0

    # T < T_freeze -> no snow melt
    T = 273.15 - 2
    ρ = 1.2
    q_sno = 1e-4
    @test snow_melt(prs, sno_prs, q_sno, ρ, T) ≈ 0

end
