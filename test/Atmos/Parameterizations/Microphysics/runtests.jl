using Test
using ClimateMachine.Microphysics_0M
using ClimateMachine.Microphysics
using Thermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics
using CLIMAParameters.Atmos.Microphysics_0M

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

@testset "0M_microphysics" begin

    _τ_precip = τ_precip(prs)
    _qc_0 = qc_0(prs)
    _S_0 = S_0(prs)

    q_vap_sat = 10e-3
    qc = 3e-3
    q_tot = 13e-3
    frac = [0.0, 0.5, 1.0]

    # no rain if no cloud
    q = PhasePartition(q_tot, 0.0, 0.0)
    @test remove_precipitation(prs, q) ≈ 0
    @test remove_precipitation(prs, q, q_vap_sat) ≈ 0

    # rain based on qc threshold
    for lf in frac
        q_liq = qc * lf
        q_ice = (1 - lf) * qc

        q = PhasePartition(q_tot, q_liq, q_ice)

        @test remove_precipitation(prs, q) ≈
              -max(0, q_liq + q_ice - _qc_0) / _τ_precip
    end

    # rain based on supersaturation threshold
    for lf in frac
        q_liq = qc * lf
        q_ice = (1 - lf) * qc

        q = PhasePartition(q_tot, q_liq, q_ice)

        @test remove_precipitation(prs, q, q_vap_sat) ≈
              -max(0, q_liq + q_ice - _S_0 * q_vap_sat) / _τ_precip
    end
end

@testset "RainFallSpeed" begin
    # eq. 5d in [Grabowski1996](@cite)
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
    q_sat_ice = q_vap_saturation_generic(prs, T, ρ, Ice())
    q = PhasePartition(q_sat_ice, 2e-3, 3e-3)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) == 0.0

    # TODO - coudnt find a plot of what it should be from the original paper
    # just chacking if the number stays the same
    T = 273.15 - 10
    q_vap = 1.02 * q_vap_saturation_generic(prs, T, ρ, Ice())
    q_liq = 0.0
    q_ice = 0.03 * q_vap
    q = PhasePartition(q_vap + q_liq + q_ice, q_liq, q_ice)
    @test conv_q_ice_to_q_sno(prs, ice_prs, q, ρ, T) ≈ 1.8512022335645584e-9

end

@testset "RainLiquidAccretion" begin

    # eq. 5b in [Grabowski1996](@cite)
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

    # eq. 5c in [Grabowski1996](@cite)
    function rain_evap_empir(
        prs::AbstractParameterSet,
        q_rai::FT,
        q::PhasePartition,
        T::FT,
        p::FT,
        ρ::FT,
    ) where {FT <: Real}

        q_sat = q_vap_saturation_generic(prs, T, ρ, Liquid())
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

@testset "hygroscopicity_test" begin

    # assumptions: standard conditions
    # parameters
    osmotic_coefficient = 3 # no units
    temperature = 298 # K
    aerosol_density = 1770 # kg/m^3
    aerosol_molecular_weight = 0.132 # kg/mol
    aerosol_particle_density = 1.0*10^-4 # 1/cm^3
    water_density = 1000 # kg/m^3
    water_particle_density = 22.414 # kg/mol
    water_molecular_weight = 0.01801528 # kg/mol

    # hand calculated value
    updraft_velocity = 3
    R = 8.31446261815324
    avogadro =  6.02214076 × 10^23

    h = (updraft_velocity * osmotic_coefficient 
        * (aerosol_particle_density * 1/avogadro 
        * aerosol_molecular_weight)/(1/water_particle_density
        * 1/1000 * water_molecular_weight) * water_molecular_weight
        * aerosol_particle_density) / (aerosol_molecular_weight
        * water_density)

    @test hygroscopicity(
               osmotic_coefficient,
               temperature,
               aerosol_density,
               aerosol_molecular_weight,
               aerosol_particle_density
               water_density,
               water_particle_density,
               water_molecular_weight,
               ) ≈ h
end 

# @testset "Mean Hygroscopicity" begin
    aerosol_component = [1, 2, 3]         
    aerosol_mode_number = [1, 2, 3, 4, 5]
    mass_mixing_ratio = [[1, 2, 3, 4, 5]
                         [0.1, 0.2, 0.3, 0.4, 0.5]
                         [0.01, 0.02, 0.03, 0.04, 0.05]]
    disassociation = [[1, 2, 3, 4, 5]
                      [0.1, 0.2, 0.3, 0.4, 0.5]
                      [0.01, 0.02, 0.03, 0.04, 0.05]]
    phi = [[1, 2, 3, 4, 5]
           [0.1, 0.2, 0.3, 0.4, 0.5]
           [0.01, 0.02, 0.03, 0.04, 0.05]]
    epsilon = [[1, 2, 3, 4, 5]
               [0.1, 0.2, 0.3, 0.4, 0.5]
               [0.01, 0.02, 0.03, 0.04, 0.05]]
    molecular_weight = [[1, 2, 3, 4, 5]
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                        [0.01, 0.02, 0.03, 0.04, 0.05]]
    add_top = 0
    add_bottom = 0
    water_molecular_weight = 0.01801528 # kg/mol
    water_density = 1000 # kg/m^3
    m_h = zeros(3)
    for i in 1:length(aerosol_mode_number)
        for j in 1:length(aerosol_component)
            add_top = mass_mixing_ratio[i][j] 
                      * disassociation[i][j] 
                      * phi[i][j]
                      * epsilon[i][j]
                      * molecular_weight[i][j]
        end
        m_h[i] = water_molecular_weight * (add_top) / (add_bottom * water_density)
    end

    @test mean_hygroscopicity(aerosol_component[1], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              phi,
                              epsilon,
                              molecular_weight,
                              ) ≈ m_h[1]
    @test mean_hygroscopicity(aerosol_component[2], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              phi,
                              epsilon,
                              molecular_weight,
                              ) ≈ m_h[2]
    @test mean_hygroscopicity(aerosol_component[3], 
                              aerosol_mode_number), 
                              mass_mixing_ratio,
                              disassociation,
                              phi,
                              epsilon,
                              molecular_weight,
                              ) ≈ m_h[3]
end


#=
    Test conditions:
        B = v, phi, epsilon, M_w, rho_a, M_a, rho_w 
        v = number of ions the salt dissociates into within water → 3
        phi = osmotic coefficient → 1
        epsilon = mass fraction of soluble material → ????
        M_w = molecular weight of water → 18.01528 g/mol
        Rho_a = density of the aerosol material → 1.77 g/cm^3
        M_a = molecular weight of aerosol material → 132
        Rho_w = density of water → 1 g/cm^3

    Test for different values of the variables
=#


# @testset "Total Number" begin
#=
    Test conditions:
        N_i = 100 cm^-3
        u_i = 
            a_ci = previous equation
            a_mi = 0.05e-6 m
            sigma_i = 2
            S_mi → dependant on a_mi
            a_mi = 0.05e-6 m
            S_max = get value from previous equation

    Test for different values of the variables
=#

# @testset "Total Mass" begin
#=
    Test conditions:
        M_i = SET URSELF
        u_i = 
            a_ci = previous equation
            a_mi = 0.05e-6 m
            sigma_i = 2
            S_mi → dependant on a_mi
            a_mi = 0.05e-6 m
            S_max = get value from previous equation


    Test for different values of the variables
=#