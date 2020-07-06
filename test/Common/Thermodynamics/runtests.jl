using Test
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using NCDatasets
using Random
using RootSolvers
MT = Thermodynamics
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Planet

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# Tolerances for tested quantities:
atol_temperature = 5e-1
atol_pressure = MSLP(param_set) * 2e-2
atol_energy = cv_d(param_set) * atol_temperature
rtol_temperature = 1e-1
rtol_density = rtol_temperature
rtol_pressure = 1e-1
rtol_energy = 1e-1

array_types = [Array{Float32}, Array{Float64}]

include("profiles.jl")
include("data_tests.jl")

@testset "Thermodynamics - isentropic processes" begin
    for ArrayType in array_types
        FT = eltype(ArrayType)

        _R_d = FT(R_d(param_set))
        _molmass_ratio = FT(molmass_ratio(param_set))
        _cp_d = FT(cp_d(param_set))
        _cp_v = FT(cp_v(param_set))
        _cp_l = FT(cp_l(param_set))
        _cv_d = FT(cv_d(param_set))
        _cv_v = FT(cv_v(param_set))
        _cv_l = FT(cv_l(param_set))
        _cv_i = FT(cv_i(param_set))
        _T_0 = FT(T_0(param_set))
        _e_int_v0 = FT(e_int_v0(param_set))
        _e_int_i0 = FT(e_int_i0(param_set))
        _LH_v0 = FT(LH_v0(param_set))
        _LH_s0 = FT(LH_s0(param_set))
        _cp_i = FT(cp_i(param_set))
        _LH_f0 = FT(LH_f0(param_set))
        _press_triple = FT(press_triple(param_set))
        _R_v = FT(R_v(param_set))
        _T_triple = FT(T_triple(param_set))
        _T_freeze = FT(T_freeze(param_set))
        _T_min = FT(T_min(param_set))
        _MSLP = FT(MSLP(param_set))
        _T_max = FT(T_max(param_set))
        _kappa_d = FT(kappa_d(param_set))

        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

        # Test state for thermodynamic consistency (with ideal gas law)
        @test all(
            T .≈
            air_temperature_from_ideal_gas_law.(Ref(param_set), p, ρ, q_pt),
        )

        Φ = FT(1)
        Random.seed!(15)
        perturbation = FT(0.1) * rand(FT, length(T))

        # TODO: Use reasonable values for ambient temperature/pressure
        T∞, p∞ = T .* perturbation, p .* perturbation
        @test air_temperature.(
            Ref(param_set),
            p,
            θ_liq_ice,
            Ref(DryAdiabaticProcess()),
        ) ≈ (p ./ _MSLP) .^ (_R_d / _cp_d) .* θ_liq_ice
        @test air_pressure_given_θ.(
            Ref(param_set),
            θ_liq_ice,
            Φ,
            Ref(DryAdiabaticProcess()),
        ) ≈ _MSLP .* (1 .- Φ ./ (θ_liq_ice .* _cp_d)) .^ (_cp_d / _R_d)
        @test air_pressure.(
            Ref(param_set),
            T,
            T∞,
            p∞,
            Ref(DryAdiabaticProcess()),
        ) ≈ p∞ .* (T ./ T∞) .^ (FT(1) / _kappa_d)
    end
end


@testset "Thermodynamics - correctness" begin
    FT = Float64
    _R_d = FT(R_d(param_set))
    _molmass_ratio = FT(molmass_ratio(param_set))
    _cp_d = FT(cp_d(param_set))
    _cp_v = FT(cp_v(param_set))
    _cp_l = FT(cp_l(param_set))
    _cv_d = FT(cv_d(param_set))
    _cv_v = FT(cv_v(param_set))
    _cv_l = FT(cv_l(param_set))
    _cv_i = FT(cv_i(param_set))
    _T_0 = FT(T_0(param_set))
    _e_int_v0 = FT(e_int_v0(param_set))
    _e_int_i0 = FT(e_int_i0(param_set))
    _LH_v0 = FT(LH_v0(param_set))
    _LH_s0 = FT(LH_s0(param_set))
    _cp_i = FT(cp_i(param_set))
    _LH_f0 = FT(LH_f0(param_set))
    _press_triple = FT(press_triple(param_set))
    _R_v = FT(R_v(param_set))
    _T_triple = FT(T_triple(param_set))
    _T_freeze = FT(T_freeze(param_set))
    _T_min = FT(T_min(param_set))
    _MSLP = FT(MSLP(param_set))
    _T_max = FT(T_max(param_set))
    _kappa_d = FT(kappa_d(param_set))
    _T_icenuc = FT(T_icenuc(param_set))

    # ideal gas law
    @test air_pressure(param_set, FT(1), FT(1), PhasePartition(FT(1))) === _R_v
    @test air_pressure(
        param_set,
        FT(1),
        FT(2),
        PhasePartition(FT(1), FT(0.5), FT(0)),
    ) === _R_v
    @test air_pressure(param_set, FT(1), FT(1)) === _R_d
    @test air_pressure(param_set, FT(1), FT(2)) === 2 * _R_d
    @test air_density(param_set, FT(1), FT(1)) === 1 / _R_d
    @test air_density(param_set, FT(1), FT(2)) === 2 / _R_d

    # gas constants and heat capacities
    @test gas_constant_air(param_set, PhasePartition(FT(0))) === _R_d
    @test gas_constant_air(param_set, PhasePartition(FT(1))) === _R_v
    @test gas_constant_air(param_set, PhasePartition(FT(0.5), FT(0.5))) ≈
          _R_d / 2
    @test gas_constant_air(param_set, FT) == _R_d

    @test cp_m(param_set, PhasePartition(FT(0))) === _cp_d
    @test cp_m(param_set, PhasePartition(FT(1))) === _cp_v
    @test cp_m(param_set, PhasePartition(FT(1), FT(1))) === _cp_l
    @test cp_m(param_set, PhasePartition(FT(1), FT(0), FT(1))) === _cp_i
    @test cp_m(param_set, FT) == _cp_d

    @test cv_m(param_set, PhasePartition(FT(0))) === _cp_d - _R_d
    @test cv_m(param_set, PhasePartition(FT(1))) === _cp_v - _R_v
    @test cv_m(param_set, PhasePartition(FT(1), FT(1))) === _cv_l
    @test cv_m(param_set, PhasePartition(FT(1), FT(0), FT(1))) === _cv_i
    @test cv_m(param_set, FT) == _cv_d

    # speed of sound
    @test soundspeed_air(param_set, _T_0 + 20, PhasePartition(FT(0))) ==
          sqrt(_cp_d / _cv_d * _R_d * (_T_0 + 20))
    @test soundspeed_air(param_set, _T_0 + 100, PhasePartition(FT(1))) ==
          sqrt(_cp_v / _cv_v * _R_v * (_T_0 + 100))

    # specific latent heats
    @test latent_heat_vapor(param_set, _T_0) ≈ _LH_v0
    @test latent_heat_fusion(param_set, _T_0) ≈ _LH_f0
    @test latent_heat_sublim(param_set, _T_0) ≈ _LH_s0

    # saturation vapor pressure and specific humidity
    p = FT(1.e5)
    q_tot = FT(0.23)
    ρ = FT(1.0)
    ρ_v_triple = _press_triple / _R_v / _T_triple
    @test saturation_vapor_pressure(param_set, _T_triple, Liquid()) ≈
          _press_triple
    @test saturation_vapor_pressure(param_set, _T_triple, Ice()) ≈ _press_triple

    phase_type = PhaseDry
    @test q_vap_saturation(
        param_set,
        _T_triple,
        ρ,
        phase_type,
        PhasePartition(FT(0)),
    ) == ρ_v_triple / ρ
    phase_type = PhaseNonEquil
    @test q_vap_saturation(
        param_set,
        _T_triple,
        ρ,
        phase_type,
        PhasePartition(q_tot, q_tot),
    ) == ρ_v_triple / ρ

    @test q_vap_saturation_generic(param_set, _T_triple, ρ; phase = Liquid()) ==
          ρ_v_triple / ρ
    @test q_vap_saturation_generic(param_set, _T_triple, ρ; phase = Ice()) ==
          ρ_v_triple / ρ
    @test q_vap_saturation_generic(
        param_set,
        _T_triple - 20,
        ρ;
        phase = Liquid(),
    ) >= q_vap_saturation_generic(param_set, _T_triple - 20, ρ; phase = Ice())

    phase_type = PhaseDry
    @test saturation_excess(
        param_set,
        _T_triple,
        ρ,
        phase_type,
        PhasePartition(q_tot),
    ) == q_tot - ρ_v_triple / ρ
    @test saturation_excess(
        param_set,
        _T_triple,
        ρ,
        phase_type,
        PhasePartition(q_tot / 1000),
    ) == 0.0

    @test supersaturation(
        param_set,
        PhasePartition(q_tot, 1e-3 * q_tot, 1e-3 * q_tot),
        ρ,
        _T_triple,
        Liquid(),
    ) ≈ 0.998 * q_tot / ρ_v_triple / ρ - 1

    @test supersaturation(
        param_set,
        PhasePartition(q_tot, 1e-3 * q_tot, 1e-3 * q_tot),
        ρ,
        _T_triple,
        Ice(),
    ) ≈ 0.998 * q_tot / ρ_v_triple / ρ - 1

    # energy functions and inverse (temperature)
    T = FT(300)
    e_kin = FT(11)
    e_pot = FT(13)
    @test air_temperature(param_set, _cv_d * (T - _T_0)) === FT(T)
    @test air_temperature(
        param_set,
        _cv_d * (T - _T_0),
        PhasePartition(FT(0)),
    ) === FT(T)

    @test air_temperature(
        param_set,
        cv_m(param_set, PhasePartition(FT(0))) * (T - _T_0),
        PhasePartition(FT(0)),
    ) === FT(T)
    @test air_temperature(
        param_set,
        cv_m(param_set, PhasePartition(FT(q_tot))) * (T - _T_0) +
        q_tot * _e_int_v0,
        PhasePartition(q_tot),
    ) ≈ FT(T)

    @test total_energy(param_set, FT(e_kin), FT(e_pot), _T_0) ===
          FT(e_kin) + FT(e_pot)
    @test total_energy(param_set, FT(e_kin), FT(e_pot), FT(T)) ≈
          FT(e_kin) + FT(e_pot) + _cv_d * (T - _T_0)
    @test total_energy(param_set, FT(0), FT(0), _T_0, PhasePartition(q_tot)) ≈
          q_tot * _e_int_v0

    # phase partitioning in equilibrium
    q_liq = FT(0.1)
    T = FT(_T_icenuc - 10)
    ρ = FT(1.0)
    q_tot = FT(0.21)
    phase_type = PhaseDry
    @test liquid_fraction(param_set, T, phase_type) === FT(0)
    phase_type = PhaseNonEquil
    @test liquid_fraction(
        param_set,
        T,
        phase_type,
        PhasePartition(q_tot, q_liq, q_liq),
    ) === FT(0.5)
    phase_type = PhaseDry
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    @test q.liq ≈ FT(0)
    @test 0 < q.ice <= q_tot

    T = FT(_T_freeze + 10)
    ρ = FT(0.1)
    q_tot = FT(0.60)
    @test liquid_fraction(param_set, T, phase_type) === FT(1)
    phase_type = PhaseNonEquil
    @test liquid_fraction(
        param_set,
        T,
        phase_type,
        PhasePartition(q_tot, q_liq, q_liq / 2),
    ) === FT(2 / 3)
    phase_type = PhaseDry
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    @test 0 < q.liq <= q_tot
    @test q.ice ≈ 0

    # saturation adjustment in equilibrium (i.e., given the thermodynamic
    # variables E_int, p, q_tot, compute the temperature and partitioning of the phases
    tol_T = 1e-1
    q_tot = FT(0)
    ρ = FT(1)
    phase_type = PhaseEquil
    @test MT.saturation_adjustment_SecantMethod(
        param_set,
        internal_energy_sat(param_set, 300.0, ρ, q_tot, phase_type),
        ρ,
        q_tot,
        phase_type,
        10,
        ResidualTolerance(1e-2),
    ) ≈ 300.0
    @test abs(
        MT.saturation_adjustment(
            param_set,
            internal_energy_sat(param_set, 300.0, ρ, q_tot, phase_type),
            ρ,
            q_tot,
            phase_type,
            10,
            ResidualTolerance(1e-2),
        ) - 300.0,
    ) < tol_T

    q_tot = FT(0.21)
    ρ = FT(0.1)
    @test MT.saturation_adjustment_SecantMethod(
        param_set,
        internal_energy_sat(param_set, 200.0, ρ, q_tot, phase_type),
        ρ,
        q_tot,
        phase_type,
        10,
        ResidualTolerance(1e-2),
    ) ≈ 200.0
    @test abs(
        MT.saturation_adjustment(
            param_set,
            internal_energy_sat(param_set, 200.0, ρ, q_tot, phase_type),
            ρ,
            q_tot,
            phase_type,
            10,
            ResidualTolerance(1e-2),
        ) - 200.0,
    ) < tol_T
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    @test q.tot - q.liq - q.ice ≈
          vapor_specific_humidity(q) ≈
          q_vap_saturation(param_set, T, ρ, phase_type)

    ρ = FT(1)
    ρu = FT[1, 2, 3]
    ρe = FT(1100)
    e_pot = FT(93)
    @test internal_energy(ρ, ρe, ρu, e_pot) ≈ 1000.0

    # potential temperatures
    T = FT(300)
    @test liquid_ice_pottemp_given_pressure(param_set, T, _MSLP) === T
    @test liquid_ice_pottemp_given_pressure(param_set, T, _MSLP / 10) ≈
          T * 10^(_R_d / _cp_d)
    @test liquid_ice_pottemp_given_pressure(
        param_set,
        T,
        _MSLP / 10,
        PhasePartition(FT(1)),
    ) ≈ T * 10^(_R_v / _cp_v)

    # dry potential temperatures. FIXME: add correctness tests
    T = FT(300)
    p = FT(1.e5)
    q_tot = FT(0.23)
    @test dry_pottemp_given_pressure(param_set, T, p, PhasePartition(q_tot)) isa
          typeof(p)
    @test air_temperature_from_liquid_ice_pottemp_given_pressure(
        param_set,
        dry_pottemp_given_pressure(param_set, T, p, PhasePartition(q_tot)),
        p,
        PhasePartition(q_tot),
    ) ≈ T

    # Exner function. FIXME: add correctness tests
    p = FT(1.e5)
    q_tot = FT(0.23)
    @test exner_given_pressure(param_set, p, PhasePartition(q_tot)) isa
          typeof(p)
end


@testset "Thermodynamics - default behavior accuracy" begin
    # Input arguments should be accurate within machine precision
    # Temperature is approximated via saturation adjustment, and should be within a physical tolerance

    for ArrayType in array_types
        FT = eltype(ArrayType)
        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

        # PhaseEquil
        ts_exact = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot, 100, FT(1e-3))
        ts = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot)
        # Should be machine accurate (because ts contains `e_int`,`ρ`,`q_tot`):
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈
            getproperty.(PhasePartition.(ts_exact), :tot),
        )
        @test all(internal_energy.(ts) .≈ internal_energy.(ts_exact))
        @test all(air_density.(ts) .≈ air_density.(ts_exact))
        # Approximate (temperature must be computed via saturation adjustment):
        @test all(isapprox.(
            air_pressure.(ts),
            air_pressure.(ts_exact),
            rtol = rtol_pressure,
        ))
        @test all(isapprox.(
            air_temperature.(ts),
            air_temperature.(ts_exact),
            rtol = rtol_temperature,
        ))

        dry_mask = abs.(q_tot .- 0) .< eps(FT)
        q_dry = q_pt[dry_mask]
        @test all(
            condensate.(q_pt) .==
            getproperty.(q_pt, :liq) .+ getproperty.(q_pt, :ice),
        )
        @test all(has_condensate.(q_dry) .== false)

        # PhaseEquil
        ts_exact =
            PhaseEquil.(
                Ref(param_set),
                e_int,
                ρ,
                q_tot,
                100,
                FT(1e-3),
                MT.saturation_adjustment_SecantMethod,
            )
        ts =
            PhaseEquil.(
                Ref(param_set),
                e_int,
                ρ,
                q_tot,
                35,
                FT(1e-1),
                MT.saturation_adjustment_SecantMethod,
            ) # Needs to be in sync with default
        # Should be machine accurate (because ts contains `e_int`,`ρ`,`q_tot`):
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈
            getproperty.(PhasePartition.(ts_exact), :tot),
        )
        @test all(internal_energy.(ts) .≈ internal_energy.(ts_exact))
        @test all(air_density.(ts) .≈ air_density.(ts_exact))
        # Approximate (temperature must be computed via saturation adjustment):
        @test all(isapprox.(
            air_pressure.(ts),
            air_pressure.(ts_exact),
            rtol = rtol_pressure,
        ))
        @test all(isapprox.(
            air_temperature.(ts),
            air_temperature.(ts_exact),
            rtol = rtol_temperature,
        ))

        # LiquidIcePotTempSHumEquil
        ts_exact =
            LiquidIcePotTempSHumEquil.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                q_tot,
                45,
                FT(1e-3),
            )
        ts = LiquidIcePotTempSHumEquil.(Ref(param_set), θ_liq_ice, ρ, q_tot)
        # Should be machine accurate:
        @test all(air_density.(ts) .≈ air_density.(ts_exact))
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈
            getproperty.(PhasePartition.(ts_exact), :tot),
        )
        # Approximate (temperature must be computed via saturation adjustment):
        @test all(isapprox.(
            internal_energy.(ts),
            internal_energy.(ts_exact),
            atol = atol_energy,
        ))
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            liquid_ice_pottemp.(ts_exact),
            rtol = rtol_temperature,
        ))
        @test all(isapprox.(
            air_temperature.(ts),
            air_temperature.(ts_exact),
            rtol = rtol_temperature,
        ))

        # LiquidIcePotTempSHumEquil_given_pressure
        ts_exact =
            LiquidIcePotTempSHumEquil_given_pressure.(
                Ref(param_set),
                θ_liq_ice,
                p,
                q_tot,
                40,
                FT(1e-3),
            )
        ts =
            LiquidIcePotTempSHumEquil_given_pressure.(
                Ref(param_set),
                θ_liq_ice,
                p,
                q_tot,
            )
        # Should be machine accurate:
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈
            getproperty.(PhasePartition.(ts_exact), :tot),
        )
        # Approximate (temperature must be computed via saturation adjustment):
        @test all(isapprox.(
            air_density.(ts),
            air_density.(ts_exact),
            rtol = rtol_density,
        ))
        @test all(isapprox.(
            internal_energy.(ts),
            internal_energy.(ts_exact),
            atol = atol_energy,
        ))
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            liquid_ice_pottemp.(ts_exact),
            rtol = rtol_temperature,
        ))
        @test all(isapprox.(
            air_temperature.(ts),
            air_temperature.(ts_exact),
            rtol = rtol_temperature,
        ))

        # LiquidIcePotTempSHumNonEquil
        ts_exact =
            LiquidIcePotTempSHumNonEquil.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                q_pt,
                40,
                FT(1e-3),
            )
        ts = LiquidIcePotTempSHumNonEquil.(Ref(param_set), θ_liq_ice, ρ, q_pt)
        # Should be machine accurate:
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈
            getproperty.(PhasePartition.(ts_exact), :tot),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :liq) .≈
            getproperty.(PhasePartition.(ts_exact), :liq),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :ice) .≈
            getproperty.(PhasePartition.(ts_exact), :ice),
        )
        @test all(air_density.(ts) .≈ air_density.(ts_exact))
        # Approximate (temperature must be computed via non-linear solve):
        @test all(isapprox.(
            internal_energy.(ts),
            internal_energy.(ts_exact),
            atol = atol_energy,
        ))
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            liquid_ice_pottemp.(ts_exact),
            rtol = rtol_temperature,
        ))
        @test all(isapprox.(
            air_temperature.(ts),
            air_temperature.(ts_exact),
            rtol = rtol_temperature,
        ))

    end

end

@testset "Thermodynamics - exceptions on failed convergence" begin

    ArrayType = Array{Float64}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

    @test_throws ErrorException MT.saturation_adjustment.(
        Ref(param_set),
        e_int,
        ρ,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException MT.saturation_adjustment_SecantMethod.(
        Ref(param_set),
        e_int,
        ρ,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    T_virt = T # should not matter: testing for non-convergence
    @test_throws ErrorException temperature_and_humidity_from_virtual_temperature.(
        Ref(param_set),
        T_virt,
        ρ,
        RH,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException air_temperature_from_liquid_ice_pottemp_non_linear.(
        Ref(param_set),
        θ_liq_ice,
        ρ,
        2,
        ResidualTolerance(FT(1e-10)),
        q_pt,
    )

    @test_throws ErrorException MT.saturation_adjustment_q_tot_θ_liq_ice.(
        Ref(param_set),
        θ_liq_ice,
        ρ,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException MT.saturation_adjustment_q_tot_θ_liq_ice_given_pressure.(
        Ref(param_set),
        θ_liq_ice,
        p,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

end

@testset "Thermodynamics - constructor consistency" begin

    # Make sure `ThermodynamicState` arguments are returned unchanged

    for ArrayType in array_types
        FT = eltype(ArrayType)
        _MSLP = FT(MSLP(param_set))

        profiles = PhaseDryProfiles(param_set, ArrayType)
        @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

        # PhaseDry
        ts = PhaseDry.(Ref(param_set), e_int, ρ)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(air_density.(ts) .≈ ρ)

        ts_p = PhaseDry_given_pT.(Ref(param_set), p, T)
        @test all(internal_energy.(ts_p) .≈ internal_energy.(Ref(param_set), T))
        @test all(air_density.(ts_p) .≈ ρ)

        ts = PhaseDry_given_ρT.(Ref(param_set), ρ, T)

        @test all(air_density.(ts_p) .≈ air_density.(ts))
        @test all(internal_energy.(ts_p) .≈ internal_energy.(ts))

        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

        # PhaseEquil
        ts =
            PhaseEquil.(
                Ref(param_set),
                e_int,
                ρ,
                q_tot,
                40,
                FT(1e-1),
                Ref(MT.saturation_adjustment_SecantMethod),
            )
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)
        @test all(air_density.(ts) .≈ ρ)

        ts = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)
        @test all(air_density.(ts) .≈ ρ)

        # PhaseNonEquil
        ts = PhaseNonEquil.(Ref(param_set), e_int, ρ, q_pt)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈ getproperty.(q_pt, :tot),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :liq) .≈ getproperty.(q_pt, :liq),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :ice) .≈ getproperty.(q_pt, :ice),
        )
        @test all(air_density.(ts) .≈ ρ)

        # air_temperature_from_liquid_ice_pottemp_given_pressure-liquid_ice_pottemp inverse
        θ_liq_ice_ =
            liquid_ice_pottemp_given_pressure.(Ref(param_set), T, p, q_pt)
        @test all(
            air_temperature_from_liquid_ice_pottemp_given_pressure.(
                Ref(param_set),
                θ_liq_ice_,
                p,
                q_pt,
            ) .≈ T,
        )

        # liquid_ice_pottemp-air_temperature_from_liquid_ice_pottemp_given_pressure inverse
        T =
            air_temperature_from_liquid_ice_pottemp_given_pressure.(
                Ref(param_set),
                θ_liq_ice,
                p,
                q_pt,
            )
        @test all(
            liquid_ice_pottemp_given_pressure.(Ref(param_set), T, p, q_pt) .≈
            θ_liq_ice,
        )

        # Accurate but expensive `LiquidIcePotTempSHumNonEquil` constructor (Non-linear temperature from θ_liq_ice)
        T_non_linear =
            air_temperature_from_liquid_ice_pottemp_non_linear.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                20,
                ResidualTolerance(FT(5e-5)),
                q_pt,
            )
        T_expansion =
            air_temperature_from_liquid_ice_pottemp.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                q_pt,
            )
        @test all(isapprox.(T_non_linear, T_expansion, rtol = rtol_temperature))
        e_int_ = internal_energy.(Ref(param_set), T_non_linear, q_pt)
        ts = PhaseNonEquil.(Ref(param_set), e_int_, ρ, q_pt)
        @test all(T_non_linear .≈ air_temperature.(ts))
        @test all(isapprox(
            θ_liq_ice,
            liquid_ice_pottemp.(ts),
            rtol = rtol_temperature,
        ))

        # LiquidIcePotTempSHumEquil
        ts =
            LiquidIcePotTempSHumEquil.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                q_tot,
                45,
                FT(1e-3),
            )
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            θ_liq_ice,
            rtol = rtol_temperature,
        ))
        @test all(isapprox.(air_density.(ts), ρ, rtol = rtol_density))
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)

        # The LiquidIcePotTempSHumEquil_given_pressure constructor
        # passes the consistency test within sufficient physical precision,
        # however, it fails to satisfy the consistency test within machine
        # precision for the input pressure.

        # LiquidIcePotTempSHumEquil_given_pressure
        ts =
            LiquidIcePotTempSHumEquil_given_pressure.(
                Ref(param_set),
                θ_liq_ice,
                p,
                q_tot,
                35,
                FT(1e-3),
            )
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            θ_liq_ice,
            rtol = rtol_temperature,
        ))
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈ getproperty.(q_pt, :tot),
        )
        @test all(isapprox.(air_pressure.(ts), p, atol = atol_pressure))

        # LiquidIcePotTempSHumNonEquil_given_pressure
        ts =
            LiquidIcePotTempSHumNonEquil_given_pressure.(
                Ref(param_set),
                θ_liq_ice,
                p,
                q_pt,
            )
        @test all(liquid_ice_pottemp.(ts) .≈ θ_liq_ice)
        @test all(air_pressure.(ts) .≈ p)
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈ getproperty.(q_pt, :tot),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :liq) .≈ getproperty.(q_pt, :liq),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :ice) .≈ getproperty.(q_pt, :ice),
        )

        # LiquidIcePotTempSHumNonEquil
        ts =
            LiquidIcePotTempSHumNonEquil.(
                Ref(param_set),
                θ_liq_ice,
                ρ,
                q_pt,
                5,
                FT(1e-3),
            )
        @test all(isapprox.(
            θ_liq_ice,
            liquid_ice_pottemp.(ts),
            rtol = rtol_temperature,
        ))
        @test all(air_density.(ts) .≈ ρ)
        @test all(
            getproperty.(PhasePartition.(ts), :tot) .≈ getproperty.(q_pt, :tot),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :liq) .≈ getproperty.(q_pt, :liq),
        )
        @test all(
            getproperty.(PhasePartition.(ts), :ice) .≈ getproperty.(q_pt, :ice),
        )


        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

        # Test that relative humidity is 1 for saturated conditions
        q_sat = q_vap_saturation.(Ref(param_set), T, ρ, Ref(phase_type))
        q_pt_sat = PhasePartition.(q_sat)
        q_vap = vapor_specific_humidity.(q_pt_sat)
        @test all(getproperty.(q_pt_sat, :liq) .≈ 0)
        @test all(getproperty.(q_pt_sat, :ice) .≈ 0)
        @test all(q_vap .≈ q_sat)

        # Compute thermodynamic consistent pressure
        p_sat = air_pressure.(Ref(param_set), T, ρ, q_pt_sat)

        # Test that density remains consistent
        ρ_rec = air_density.(Ref(param_set), T, p_sat, q_pt_sat)
        @test all.(ρ_rec ≈ ρ)

        RH_sat =
            relative_humidity.(
                Ref(param_set),
                T,
                p_sat,
                Ref(phase_type),
                q_pt_sat,
            )

        # TODO: Add this test back in
        @test all(RH_sat .≈ 1)

        # Test that RH is zero for dry conditions
        q_pt_dry = PhasePartition.(zeros(FT, length(T)))
        p_dry = air_pressure.(Ref(param_set), T, ρ, q_pt_dry)
        RH_dry =
            relative_humidity.(
                Ref(param_set),
                T,
                p_dry,
                Ref(phase_type),
                q_pt_dry,
            )
        @test all(RH_dry .≈ 0)


        # Test virtual temperature and inverse functions:
        _R_d = FT(R_d(param_set))
        T_virt = virtual_temperature.(Ref(param_set), T, ρ, q_pt)
        @test all(T_virt ≈ gas_constant_air.(Ref(param_set), q_pt) ./ _R_d .* T)

        T_rec_qpt_rec =
            temperature_and_humidity_from_virtual_temperature.(
                Ref(param_set),
                T_virt,
                ρ,
                RH,
                Ref(phase_type),
            )

        T_rec = first.(T_rec_qpt_rec)
        q_pt_rec = last.(T_rec_qpt_rec)

        # Test convergence of virtual temperature iterations
        @test all(isapprox.(
            T_virt,
            virtual_temperature.(Ref(param_set), T_rec, ρ, q_pt_rec),
            atol = sqrt(eps(FT)),
        ))

        # Test that reconstructed specific humidity is close
        # to original specific humidity
        q_tot_rec = getproperty.(q_pt_rec, :tot)
        RH_moist = q_tot .> eps(FT)
        @test all(isapprox.(q_tot[RH_moist], q_tot_rec[RH_moist], rtol = 5e-2))

        # Update temperature to be exactly consistent with
        # p, ρ, q_pt_rec; test that this is equal to T_rec
        T_local =
            air_temperature_from_ideal_gas_law.(Ref(param_set), p, ρ, q_pt_rec)
        @test all(isapprox.(T_local, T_rec, atol = sqrt(eps(FT))))
    end

end


@testset "Thermodynamics - type-stability" begin

    # NOTE: `Float32` saturation adjustment tends to have more difficulty
    # with converging to the same tolerances as `Float64`, so they're relaxed here.
    ArrayType = Array{Float32}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

    ρu = FT[1.0, 2.0, 3.0]
    e_pot = FT(100.0)
    @test typeof.(internal_energy.(ρ, ρ .* e_int, Ref(ρu), Ref(e_pot))) ==
          typeof.(e_int)

    ts_dry = PhaseDry.(Ref(param_set), e_int, ρ)
    ts_dry_pT = PhaseDry_given_pT.(Ref(param_set), p, T)
    ts_eq = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot, 15, FT(1e-1))

    ts_T =
        TemperatureSHumEquil.(
            Ref(param_set),
            air_temperature.(ts_dry),
            air_density.(ts_dry),
            q_tot,
        )
    ts_Tp =
        TemperatureSHumEquil_given_pressure.(
            Ref(param_set),
            air_temperature.(ts_dry),
            air_pressure.(ts_dry),
            q_tot,
        )

    @test all(air_temperature.(ts_T) .≈ air_temperature.(ts_Tp))
    # @test all(isapprox.(air_pressure.(ts_T), air_pressure.(ts_Tp), atol = _MSLP * 2e-2)) # TODO: Fails, needs fixing / better test
    @test all(total_specific_humidity.(ts_T) .≈ total_specific_humidity.(ts_Tp))

    ts_neq = PhaseNonEquil.(Ref(param_set), e_int, ρ, q_pt)
    ts_T_neq = TemperatureSHumNonEquil.(Ref(param_set), T, ρ, q_pt)

    ts_θ_liq_ice_eq =
        LiquidIcePotTempSHumEquil.(
            Ref(param_set),
            θ_liq_ice,
            ρ,
            q_tot,
            45,
            FT(1e-3),
        )
    ts_θ_liq_ice_eq_p =
        LiquidIcePotTempSHumEquil_given_pressure.(
            Ref(param_set),
            θ_liq_ice,
            p,
            q_tot,
            40,
            FT(1e-3),
        )
    ts_θ_liq_ice_neq =
        LiquidIcePotTempSHumNonEquil.(Ref(param_set), θ_liq_ice, ρ, q_pt)
    ts_θ_liq_ice_neq_p =
        LiquidIcePotTempSHumNonEquil_given_pressure.(
            Ref(param_set),
            θ_liq_ice,
            p,
            q_pt,
        )

    for ts in (
        ts_dry,
        ts_dry_pT,
        ts_eq,
        ts_T,
        ts_Tp,
        ts_neq,
        ts_T_neq,
        ts_θ_liq_ice_eq,
        ts_θ_liq_ice_eq_p,
        ts_θ_liq_ice_neq,
        ts_θ_liq_ice_neq_p,
    )
        @test typeof.(soundspeed_air.(ts)) == typeof.(e_int)
        @test typeof.(gas_constant_air.(ts)) == typeof.(e_int)
        @test typeof.(vapor_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(relative_humidity.(ts)) == typeof.(e_int)
        @test typeof.(air_pressure.(ts)) == typeof.(e_int)
        @test typeof.(air_density.(ts)) == typeof.(e_int)
        @test typeof.(total_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(cp_m.(ts)) == typeof.(e_int)
        @test typeof.(cv_m.(ts)) == typeof.(e_int)
        @test typeof.(air_temperature.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_sat.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_vapor.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_sublim.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_fusion.(ts)) == typeof.(e_int)
        @test typeof.(q_vap_saturation.(ts)) == typeof.(e_int)
        @test typeof.(saturation_excess.(ts)) == typeof.(e_int)
        @test typeof.(liquid_fraction.(ts)) == typeof.(e_int)
        @test typeof.(liquid_ice_pottemp.(ts)) == typeof.(e_int)
        @test typeof.(dry_pottemp.(ts)) == typeof.(e_int)
        @test typeof.(exner.(ts)) == typeof.(e_int)
        @test typeof.(liquid_ice_pottemp_sat.(ts)) == typeof.(e_int)
        @test typeof.(specific_volume.(ts)) == typeof.(e_int)
        @test typeof.(virtual_pottemp.(ts)) == typeof.(e_int)
        @test eltype.(gas_constants.(ts)) == typeof.(e_int)
        @test typeof.(getproperty.(PhasePartition.(ts), :tot)) == typeof.(e_int)
    end

end

@testset "Thermodynamics - dry limit" begin

    ArrayType = Array{Float64}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack_fields profiles T p RS e_int ρ θ_liq_ice q_tot q_liq q_ice q_pt RH phase_type

    # PhasePartition test is noisy, so do this only once:
    ts_dry = PhaseDry(param_set, first(e_int), first(ρ))
    ts_eq = PhaseEquil(param_set, first(e_int), first(ρ), typeof(first(ρ))(0))
    @test PhasePartition(ts_eq).tot ≈ PhasePartition(ts_dry).tot
    @test PhasePartition(ts_eq).liq ≈ PhasePartition(ts_dry).liq
    @test PhasePartition(ts_eq).ice ≈ PhasePartition(ts_dry).ice

    ts_dry = PhaseDry.(Ref(param_set), e_int, ρ)
    ts_eq = PhaseEquil.(Ref(param_set), e_int, ρ, q_tot .* 0)

    @test all(gas_constant_air.(ts_eq) .≈ gas_constant_air.(ts_dry))
    @test all(relative_humidity.(ts_eq) .≈ relative_humidity.(ts_dry))
    @test all(air_pressure.(ts_eq) .≈ air_pressure.(ts_dry))
    @test all(air_density.(ts_eq) .≈ air_density.(ts_dry))
    @test all(specific_volume.(ts_eq) .≈ specific_volume.(ts_dry))
    @test all(
        total_specific_humidity.(ts_eq) .≈ total_specific_humidity.(ts_dry),
    )
    @test all(cp_m.(ts_eq) .≈ cp_m.(ts_dry))
    @test all(cv_m.(ts_eq) .≈ cv_m.(ts_dry))
    @test all(air_temperature.(ts_eq) .≈ air_temperature.(ts_dry))
    @test all(internal_energy.(ts_eq) .≈ internal_energy.(ts_dry))
    @test all(internal_energy_sat.(ts_eq) .≈ internal_energy_sat.(ts_dry))
    @test all(soundspeed_air.(ts_eq) .≈ soundspeed_air.(ts_dry))
    @test all(latent_heat_vapor.(ts_eq) .≈ latent_heat_vapor.(ts_dry))
    @test all(latent_heat_sublim.(ts_eq) .≈ latent_heat_sublim.(ts_dry))
    @test all(latent_heat_fusion.(ts_eq) .≈ latent_heat_fusion.(ts_dry))
    @test all(q_vap_saturation.(ts_eq) .≈ q_vap_saturation.(ts_dry))
    @test all(saturation_excess.(ts_eq) .≈ saturation_excess.(ts_dry))
    @test all(liquid_fraction.(ts_eq) .≈ liquid_fraction.(ts_dry))
    @test all(liquid_ice_pottemp.(ts_eq) .≈ liquid_ice_pottemp.(ts_dry))
    @test all(dry_pottemp.(ts_eq) .≈ dry_pottemp.(ts_dry))
    @test all(virtual_pottemp.(ts_eq) .≈ virtual_pottemp.(ts_dry))
    @test all(liquid_ice_pottemp_sat.(ts_eq) .≈ liquid_ice_pottemp_sat.(ts_dry))
    @test all(exner.(ts_eq) .≈ exner.(ts_dry))

    @test all(
        saturation_vapor_pressure.(ts_eq, Ref(Ice())) .≈
        saturation_vapor_pressure.(ts_dry, Ref(Ice())),
    )
    @test all(
        saturation_vapor_pressure.(ts_eq, Ref(Liquid())) .≈
        saturation_vapor_pressure.(ts_dry, Ref(Liquid())),
    )
    @test all(first.(gas_constants.(ts_eq)) ≈ first.(gas_constants.(ts_dry)))
    @test all(last.(gas_constants.(ts_eq)) ≈ last.(gas_constants.(ts_dry)))

end
