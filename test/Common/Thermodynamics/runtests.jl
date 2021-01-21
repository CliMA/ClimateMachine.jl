module TestThermodynamics
using Test
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using UnPack
using BenchmarkTools
using NCDatasets
using Random
using RootSolvers
const TD = Thermodynamics
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

compare_moisture(a::ThermodynamicState, b::ThermodynamicState) =
    compare_moisture(a, PhasePartition(b))

compare_moisture(ts::PhaseEquil, q_pt::PhasePartition) =
    getproperty(PhasePartition(ts), :tot) ≈ getproperty(q_pt, :tot)

compare_moisture(ts::PhaseNonEquil, q_pt::PhasePartition) = all((
    getproperty(PhasePartition(ts), :tot) ≈ getproperty(q_pt, :tot),
    getproperty(PhasePartition(ts), :liq) ≈ getproperty(q_pt, :liq),
    getproperty(PhasePartition(ts), :ice) ≈ getproperty(q_pt, :ice),
))

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
        @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
        @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

        # Test state for thermodynamic consistency (with ideal gas law)
        T_idgl = TD.air_temperature_from_ideal_gas_law.(param_set, p, ρ, q_pt)
        @test all(T .≈ T_idgl)

        Φ = FT(1)
        Random.seed!(15)
        perturbation = FT(0.1) * rand(FT, length(T))

        # TODO: Use reasonable values for ambient temperature/pressure
        T∞, p∞ = T .* perturbation, p .* perturbation
        @test air_temperature.(param_set, p, θ_liq_ice, DryAdiabaticProcess()) ≈
              (p ./ _MSLP) .^ (_R_d / _cp_d) .* θ_liq_ice
        @test TD.air_pressure_given_θ.(
            param_set,
            θ_liq_ice,
            Φ,
            DryAdiabaticProcess(),
        ) ≈ _MSLP .* (1 .- Φ ./ (θ_liq_ice .* _cp_d)) .^ (_cp_d / _R_d)
        @test air_pressure.(param_set, T, T∞, p∞, DryAdiabaticProcess()) ≈
              p∞ .* (T ./ T∞) .^ (FT(1) / _kappa_d)
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

    @test q_vap_saturation_generic(param_set, _T_triple, ρ, Liquid()) ==
          ρ_v_triple / ρ
    @test q_vap_saturation_generic(param_set, _T_triple, ρ, Ice()) ==
          ρ_v_triple / ρ
    @test q_vap_saturation_generic(param_set, _T_triple - 20, ρ, Liquid()) >=
          q_vap_saturation_generic(param_set, _T_triple - 20, ρ, Ice())

    # test the wrapper for q_vap_saturation over liquid water and ice
    ρ = FT(1)
    ρu = FT[1, 2, 3]
    ρe = FT(1100)
    e_pot = FT(93)
    e_int = internal_energy(ρ, ρe, ρu, e_pot)
    q_pt = PhasePartition(FT(0.02), FT(0.002), FT(0.002))
    ts = PhaseNonEquil(param_set, e_int, ρ, q_pt)
    @test q_vap_saturation_generic(
        param_set,
        air_temperature(ts),
        ρ,
        Liquid(),
    ) ≈ q_vap_saturation_liquid(ts)
    @test q_vap_saturation_generic(param_set, air_temperature(ts), ρ, Ice()) ≈
          q_vap_saturation_ice(ts)
    @test q_vap_saturation_ice(ts) <= q_vap_saturation_liquid(ts)

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
    q_tot = FT(0)
    ρ = FT(1)
    phase_type = PhaseEquil
    @test TD.saturation_adjustment(
        SecantMethod,
        param_set,
        internal_energy_sat(param_set, 300.0, ρ, q_tot, phase_type),
        ρ,
        q_tot,
        phase_type,
        10,
        rtol_temperature,
    ) ≈ 300.0
    @test abs(
        TD.saturation_adjustment(
            NewtonsMethod,
            param_set,
            internal_energy_sat(param_set, 300.0, ρ, q_tot, phase_type),
            ρ,
            q_tot,
            phase_type,
            10,
            rtol_temperature,
        ) - 300.0,
    ) < rtol_temperature

    q_tot = FT(0.21)
    ρ = FT(0.1)
    @test isapprox(
        TD.saturation_adjustment(
            SecantMethod,
            param_set,
            internal_energy_sat(param_set, 200.0, ρ, q_tot, phase_type),
            ρ,
            q_tot,
            phase_type,
            10,
            rtol_temperature,
        ),
        200.0,
        rtol = rtol_temperature,
    )
    @test abs(
        TD.saturation_adjustment(
            NewtonsMethod,
            param_set,
            internal_energy_sat(param_set, 200.0, ρ, q_tot, phase_type),
            ρ,
            q_tot,
            phase_type,
            10,
            rtol_temperature,
        ) - 200.0,
    ) < rtol_temperature
    q = PhasePartition_equil(param_set, T, ρ, q_tot, phase_type)
    @test q.tot - q.liq - q.ice ≈
          vapor_specific_humidity(q) ≈
          q_vap_saturation(param_set, T, ρ, phase_type)

    ρ = FT(1)
    ρu = FT[1, 2, 3]
    ρe = FT(1100)
    e_pot = FT(93)
    @test internal_energy(ρ, ρe, ρu, e_pot) ≈ 1000.0

    # internal energies for dry, vapor, liquid and ice
    T = FT(300)
    q = PhasePartition(FT(20 * 1e-3), FT(5 * 1e-3), FT(2 * 1e-3))
    q_vap = vapor_specific_humidity(q)
    Id = internal_energy_dry(param_set, T)
    Iv = internal_energy_vapor(param_set, T)
    Il = internal_energy_liquid(param_set, T)
    Ii = internal_energy_ice(param_set, T)
    @test internal_energy(param_set, T, q) ≈
          (1 - q.tot) * Id + q_vap * Iv + q.liq * Il + q.ice * Ii
    @test internal_energy(param_set, T) ≈ Id

    # potential temperatures
    T = FT(300)
    @test TD.liquid_ice_pottemp_given_pressure(param_set, T, _MSLP) === T
    @test TD.liquid_ice_pottemp_given_pressure(param_set, T, _MSLP / 10) ≈
          T * 10^(_R_d / _cp_d)
    @test TD.liquid_ice_pottemp_given_pressure(
        param_set,
        T,
        _MSLP / 10,
        PhasePartition(FT(1)),
    ) ≈ T * 10^(_R_v / _cp_v)

    # dry potential temperatures. FIXME: add correctness tests
    T = FT(300)
    p = FT(1.e5)
    q_tot = FT(0.23)
    @test TD.dry_pottemp_given_pressure(
        param_set,
        T,
        p,
        PhasePartition(q_tot),
    ) isa typeof(p)
    @test TD.air_temperature_given_θpq(
        param_set,
        TD.dry_pottemp_given_pressure(param_set, T, p, PhasePartition(q_tot)),
        p,
        PhasePartition(q_tot),
    ) ≈ T

    # Exner function. FIXME: add correctness tests
    p = FT(1.e5)
    q_tot = FT(0.23)
    @test TD.exner_given_pressure(param_set, p, PhasePartition(q_tot)) isa
          typeof(p)

    q_tot = 0.1
    q_liq = 0.05
    q_ice = 0.01
    mr = shum_to_mixing_ratio(q_tot, q_tot)
    @test mr == q_tot / (1 - q_tot)
    mr = shum_to_mixing_ratio(q_liq, q_tot)
    @test mr == q_liq / (1 - q_tot)

    q = PhasePartition(q_tot, q_liq, q_ice)
    mrs = mixing_ratios(q)
    @test mrs.tot == q_tot / (1 - q_tot)
    @test mrs.liq == q_liq / (1 - q_tot)
    @test mrs.ice == q_ice / (1 - q_tot)
end


@testset "Thermodynamics - default behavior accuracy" begin
    # Input arguments should be accurate within machine precision
    # Temperature is approximated via saturation adjustment, and should be within a physical tolerance

    or(a, b) = a || b
    for ArrayType in array_types
        FT = eltype(ArrayType)
        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
        @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

        RH_sat_mask = or.(RH .> 1, RH .≈ 1)
        RH_unsat_mask = .!or.(RH .> 1, RH .≈ 1)
        ts = PhaseEquil.(param_set, e_int, ρ, q_tot)
        @test all(saturated.(ts[RH_sat_mask]))
        @test !any(saturated.(ts[RH_unsat_mask]))

        # PhaseEquil (freezing)
        _T_freeze = FT(T_freeze(param_set))
        e_int_upper =
            internal_energy_sat.(
                param_set,
                Ref(_T_freeze + sqrt(eps(FT))),
                ρ,
                q_tot,
                phase_type,
            )
        e_int_lower =
            internal_energy_sat.(
                param_set,
                Ref(_T_freeze - sqrt(eps(FT))),
                ρ,
                q_tot,
                phase_type,
            )
        _e_int = (e_int_upper .+ e_int_lower) / 2
        ts = PhaseEquil.(param_set, _e_int, ρ, q_tot)
        @test all(air_temperature.(ts) .== Ref(_T_freeze))

        # Args needs to be in sync with PhaseEquil:
        ts = PhaseEquil.(param_set, _e_int, ρ, q_tot, 8, FT(1e-1), SecantMethod)
        @test all(air_temperature.(ts) .== Ref(_T_freeze))

        # PhaseEquil
        ts_exact = PhaseEquil.(param_set, e_int, ρ, q_tot, 100, FT(1e-3))
        ts = PhaseEquil.(param_set, e_int, ρ, q_tot)
        @test all(isapprox.(T, air_temperature.(ts), rtol = rtol_temperature))

        # Should be machine accurate (because ts contains `e_int`,`ρ`,`q_tot`):
        @test all(compare_moisture.(ts, ts_exact))
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

        e_tot = total_energy.(e_kin, e_pot, ts)
        @test all(
            specific_enthalpy.(ts) .≈
            e_int .+ gas_constant_air.(ts) .* air_temperature.(ts),
        )
        @test all(
            total_specific_enthalpy.(ts, e_tot) .≈
            specific_enthalpy.(ts) .+ e_kin .+ e_pot,
        )
        @test all(
            moist_static_energy.(ts, e_pot) .≈ specific_enthalpy.(ts) .+ e_pot,
        )
        @test all(
            moist_static_energy.(ts, e_pot) .≈
            total_specific_enthalpy.(ts, e_tot) .- e_kin,
        )

        # PhaseEquil
        ts_exact =
            PhaseEquil.(param_set, e_int, ρ, q_tot, 100, FT(1e-3), SecantMethod)
        ts = PhaseEquil.(param_set, e_int, ρ, q_tot, 35, FT(1e-1), SecantMethod) # Needs to be in sync with default
        # Should be machine accurate (because ts contains `e_int`,`ρ`,`q_tot`):
        @test all(compare_moisture.(ts, ts_exact))
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

        # PhaseEquil_ρθq
        ts_exact = PhaseEquil_ρθq.(param_set, ρ, θ_liq_ice, q_tot, 45, FT(1e-3))
        ts = PhaseEquil_ρθq.(param_set, ρ, θ_liq_ice, q_tot)
        # Should be machine accurate:
        @test all(air_density.(ts) .≈ air_density.(ts_exact))
        @test all(compare_moisture.(ts, ts_exact))
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

        # PhaseEquil_pθq
        ts_exact = PhaseEquil_pθq.(param_set, p, θ_liq_ice, q_tot, 40, FT(1e-3))
        ts = PhaseEquil_pθq.(param_set, p, θ_liq_ice, q_tot)
        # Should be machine accurate:
        @test all(compare_moisture.(ts, ts_exact))
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

        # PhaseNonEquil_ρθq
        ts_exact =
            PhaseNonEquil_ρθq.(param_set, ρ, θ_liq_ice, q_pt, 40, FT(1e-3))
        ts = PhaseNonEquil_ρθq.(param_set, ρ, θ_liq_ice, q_pt)
        # Should be machine accurate:
        @test all(compare_moisture.(ts, ts_exact))
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
    @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
    @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

    @test_throws ErrorException TD.saturation_adjustment.(
        NewtonsMethod,
        param_set,
        e_int,
        ρ,
        q_tot,
        Ref(phase_type),
        2,
        FT(1e-10),
    )

    @test_throws ErrorException TD.saturation_adjustment.(
        SecantMethod,
        param_set,
        e_int,
        ρ,
        q_tot,
        Ref(phase_type),
        2,
        FT(1e-10),
    )

    T_virt = T # should not matter: testing for non-convergence
    @test_throws ErrorException TD.temperature_and_humidity_given_TᵥρRH.(
        param_set,
        T_virt,
        ρ,
        RH,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException TD.air_temperature_given_θρq_nonlinear.(
        param_set,
        θ_liq_ice,
        ρ,
        2,
        ResidualTolerance(FT(1e-10)),
        q_pt,
    )

    @test_throws ErrorException TD.saturation_adjustment_given_ρθq.(
        param_set,
        ρ,
        θ_liq_ice,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException TD.saturation_adjustment_given_pθq.(
        param_set,
        p,
        θ_liq_ice,
        q_tot,
        Ref(phase_type),
        2,
        ResidualTolerance(FT(1e-10)),
    )

    @test_throws ErrorException TD.saturation_adjustment_ρpq.(
        NewtonsMethodAD,
        param_set,
        ρ,
        p,
        q_tot,
        Ref(phase_type),
        2,
        FT(1e-10),
    )

end

@testset "Thermodynamics - constructor consistency" begin

    # Make sure `ThermodynamicState` arguments are returned unchanged

    for ArrayType in array_types
        FT = eltype(ArrayType)
        _MSLP = FT(MSLP(param_set))

        profiles = PhaseDryProfiles(param_set, ArrayType)
        @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
        @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

        # PhaseDry
        ts = PhaseDry.(param_set, e_int, ρ)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(air_density.(ts) .≈ ρ)

        ts_pT = PhaseDry_pT.(param_set, p, T)
        @test all(internal_energy.(ts_pT) .≈ internal_energy.(param_set, T))
        @test all(air_density.(ts_pT) .≈ ρ)

        θ_dry = dry_pottemp.(param_set, T, ρ)
        ts_pθ = PhaseDry_pθ.(param_set, p, θ_dry)
        @test all(internal_energy.(ts_pθ) .≈ internal_energy.(param_set, T))
        @test all(air_density.(ts_pθ) .≈ ρ)

        ts_ρθ = PhaseDry_ρθ.(param_set, ρ, θ_dry)
        @test all(internal_energy.(ts_ρθ) .≈ internal_energy.(param_set, T))
        @test all(air_density.(ts_ρθ) .≈ ρ)

        ts_ρT = PhaseDry_ρT.(param_set, ρ, T)
        @test all(air_density.(ts_ρT) .≈ air_density.(ts))
        @test all(internal_energy.(ts_ρT) .≈ internal_energy.(ts))


        ts = PhaseDry_ρp.(param_set, ρ, p)
        @test all(air_density.(ts) .≈ ρ)
        @test all(air_pressure.(ts) .≈ p)
        e_tot_proposed =
            TD.total_energy_given_ρp.(param_set, ρ, p, e_kin, e_pot)
        @test all(total_energy.(e_kin, e_pot, ts) .≈ e_tot_proposed)


        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
        @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

        # PhaseEquil
        ts = PhaseEquil.(param_set, e_int, ρ, q_tot, 40, FT(1e-1), SecantMethod)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)
        @test all(air_density.(ts) .≈ ρ)

        ts = PhaseEquil.(param_set, e_int, ρ, q_tot)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)
        @test all(air_density.(ts) .≈ ρ)

        ts = PhaseEquil_ρpq.(param_set, ρ, p, q_tot, true)
        @test all(air_density.(ts) .≈ ρ)
        @test all(air_pressure.(ts) .≈ p)
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)

        # Test against total_energy_given_ρp when not iterating
        ts = PhaseEquil_ρpq.(param_set, ρ, p, q_tot, false)
        e_tot_proposed =
            TD.total_energy_given_ρp.(
                param_set,
                ρ,
                p,
                e_kin,
                e_pot,
                PhasePartition.(q_tot),
            )
        @test all(total_energy.(e_kin, e_pot, ts) .≈ e_tot_proposed)

        # PhaseNonEquil
        ts = PhaseNonEquil.(param_set, e_int, ρ, q_pt)
        @test all(internal_energy.(ts) .≈ e_int)
        @test all(compare_moisture.(ts, q_pt))
        @test all(air_density.(ts) .≈ ρ)

        # TD.air_temperature_given_θpq-liquid_ice_pottemp inverse
        θ_liq_ice_ =
            TD.liquid_ice_pottemp_given_pressure.(param_set, T, p, q_pt)
        @test all(
            TD.air_temperature_given_θpq.(param_set, θ_liq_ice_, p, q_pt) .≈ T,
        )

        # liquid_ice_pottemp-TD.air_temperature_given_θpq inverse
        T = TD.air_temperature_given_θpq.(param_set, θ_liq_ice, p, q_pt)
        @test all(
            TD.liquid_ice_pottemp_given_pressure.(param_set, T, p, q_pt) .≈
            θ_liq_ice,
        )

        # Accurate but expensive `PhaseNonEquil_ρθq` constructor (Non-linear temperature from θ_liq_ice)
        T_non_linear =
            TD.air_temperature_given_θρq_nonlinear.(
                param_set,
                θ_liq_ice,
                ρ,
                20,
                ResidualTolerance(FT(5e-5)),
                q_pt,
            )
        T_expansion =
            TD.air_temperature_given_θρq.(param_set, θ_liq_ice, ρ, q_pt)
        @test all(isapprox.(T_non_linear, T_expansion, rtol = rtol_temperature))
        e_int_ = internal_energy.(param_set, T_non_linear, q_pt)
        ts = PhaseNonEquil.(param_set, e_int_, ρ, q_pt)
        @test all(T_non_linear .≈ air_temperature.(ts))
        @test all(isapprox(
            θ_liq_ice,
            liquid_ice_pottemp.(ts),
            rtol = rtol_temperature,
        ))

        # PhaseEquil_ρθq
        ts = PhaseEquil_ρθq.(param_set, ρ, θ_liq_ice, q_tot, 45, FT(1e-3))
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            θ_liq_ice,
            rtol = rtol_temperature,
        ))
        @test all(isapprox.(air_density.(ts), ρ, rtol = rtol_density))
        @test all(getproperty.(PhasePartition.(ts), :tot) .≈ q_tot)

        # The PhaseEquil_pθq constructor
        # passes the consistency test within sufficient physical precision,
        # however, it fails to satisfy the consistency test within machine
        # precision for the input pressure.

        # PhaseEquil_pθq
        ts = PhaseEquil_pθq.(param_set, p, θ_liq_ice, q_tot, 35, FT(1e-3))
        @test all(isapprox.(
            liquid_ice_pottemp.(ts),
            θ_liq_ice,
            rtol = rtol_temperature,
        ))
        @test all(compare_moisture.(ts, q_pt))
        @test all(isapprox.(air_pressure.(ts), p, atol = atol_pressure))

        # PhaseNonEquil_pθq
        ts = PhaseNonEquil_pθq.(param_set, p, θ_liq_ice, q_pt)
        @test all(liquid_ice_pottemp.(ts) .≈ θ_liq_ice)
        @test all(air_pressure.(ts) .≈ p)
        @test all(compare_moisture.(ts, q_pt))

        ts = PhaseNonEquil_ρpq.(param_set, ρ, p, q_pt)
        @test all(air_density.(ts) .≈ ρ)
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
        e_tot_proposed =
            TD.total_energy_given_ρp.(param_set, ρ, p, e_kin, e_pot, q_pt)
        @test all(total_energy.(e_kin, e_pot, ts) .≈ e_tot_proposed)

        # PhaseNonEquil_ρθq
        ts = PhaseNonEquil_ρθq.(param_set, ρ, θ_liq_ice, q_pt, 5, FT(1e-3))
        @test all(isapprox.(
            θ_liq_ice,
            liquid_ice_pottemp.(ts),
            rtol = rtol_temperature,
        ))
        @test all(air_density.(ts) .≈ ρ)
        @test all(compare_moisture.(ts, q_pt))

        profiles = PhaseEquilProfiles(param_set, ArrayType)
        @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
        @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

        # Test that relative humidity is 1 for saturated conditions
        q_sat = q_vap_saturation.(param_set, T, ρ, Ref(phase_type))
        q_pt_sat = PhasePartition.(q_sat)
        q_vap = vapor_specific_humidity.(q_pt_sat)
        @test all(getproperty.(q_pt_sat, :liq) .≈ 0)
        @test all(getproperty.(q_pt_sat, :ice) .≈ 0)
        @test all(q_vap .≈ q_sat)

        # Compute thermodynamic consistent pressure
        p_sat = air_pressure.(param_set, T, ρ, q_pt_sat)

        # Test that density remains consistent
        ρ_rec = air_density.(param_set, T, p_sat, q_pt_sat)
        @test all.(ρ_rec ≈ ρ)

        RH_sat =
            relative_humidity.(param_set, T, p_sat, Ref(phase_type), q_pt_sat)

        # TODO: Add this test back in
        @test all(RH_sat .≈ 1)

        # Test that RH is zero for dry conditions
        q_pt_dry = PhasePartition.(zeros(FT, length(T)))
        p_dry = air_pressure.(param_set, T, ρ, q_pt_dry)
        RH_dry =
            relative_humidity.(param_set, T, p_dry, Ref(phase_type), q_pt_dry)
        @test all(RH_dry .≈ 0)


        # Test virtual temperature and inverse functions:
        _R_d = FT(R_d(param_set))
        T_virt = virtual_temperature.(param_set, T, ρ, q_pt)
        @test all(T_virt ≈ gas_constant_air.(param_set, q_pt) ./ _R_d .* T)

        T_rec_qpt_rec =
            TD.temperature_and_humidity_given_TᵥρRH.(
                param_set,
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
            virtual_temperature.(param_set, T_rec, ρ, q_pt_rec),
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
            TD.air_temperature_from_ideal_gas_law.(param_set, p, ρ, q_pt_rec)
        @test all(isapprox.(T_local, T_rec, atol = sqrt(eps(FT))))
    end

end


@testset "Thermodynamics - type-stability" begin

    # NOTE: `Float32` saturation adjustment tends to have more difficulty
    # with converging to the same tolerances as `Float64`, so they're relaxed here.
    ArrayType = Array{Float32}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
    @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

    ρu = FT[1.0, 2.0, 3.0]
    @test typeof.(internal_energy.(ρ, ρ .* e_int, Ref(ρu), e_pot)) ==
          typeof.(e_int)

    θ_dry = dry_pottemp.(param_set, T, ρ)
    ts_dry = PhaseDry.(param_set, e_int, ρ)
    ts_dry_ρp = PhaseDry_ρp.(param_set, ρ, p)
    ts_dry_pT = PhaseDry_pT.(param_set, p, T)
    ts_dry_ρθ = PhaseDry_ρθ.(param_set, ρ, θ_dry)
    ts_dry_pθ = PhaseDry_pθ.(param_set, p, θ_dry)
    ts_eq = PhaseEquil.(param_set, e_int, ρ, q_tot, 15, FT(1e-1))
    e_tot = total_energy.(e_kin, e_pot, ts_eq)

    ts_T =
        PhaseEquil_ρTq.(
            param_set,
            air_density.(ts_dry),
            air_temperature.(ts_dry),
            q_tot,
        )
    ts_Tp =
        PhaseEquil_pTq.(
            param_set,
            air_pressure.(ts_dry),
            air_temperature.(ts_dry),
            q_tot,
        )

    ts_ρp =
        PhaseEquil_ρpq.(
            param_set,
            air_density.(ts_dry),
            air_pressure.(ts_dry),
            q_tot,
        )

    @test all(air_temperature.(ts_T) .≈ air_temperature.(ts_Tp))
    # @test all(isapprox.(air_pressure.(ts_T), air_pressure.(ts_Tp), atol = _MSLP * 2e-2)) # TODO: Fails, needs fixing / better test
    @test all(total_specific_humidity.(ts_T) .≈ total_specific_humidity.(ts_Tp))

    ts_neq = PhaseNonEquil.(param_set, e_int, ρ, q_pt)
    ts_T_neq = PhaseNonEquil_ρTq.(param_set, ρ, T, q_pt)

    ts_θ_liq_ice_eq =
        PhaseEquil_ρθq.(param_set, ρ, θ_liq_ice, q_tot, 45, FT(1e-3))
    ts_θ_liq_ice_eq_p =
        PhaseEquil_pθq.(param_set, p, θ_liq_ice, q_tot, 40, FT(1e-3))
    ts_θ_liq_ice_neq = PhaseNonEquil_ρθq.(param_set, ρ, θ_liq_ice, q_pt)
    ts_θ_liq_ice_neq_p = PhaseNonEquil_pθq.(param_set, p, θ_liq_ice, q_pt)

    for ts in (
        ts_dry,
        ts_dry_ρp,
        ts_dry_pT,
        ts_dry_ρθ,
        ts_dry_pθ,
        ts_eq,
        ts_T,
        ts_Tp,
        ts_ρp,
        ts_neq,
        ts_T_neq,
        ts_θ_liq_ice_eq,
        ts_θ_liq_ice_eq_p,
        ts_θ_liq_ice_neq,
        ts_θ_liq_ice_neq_p,
    )
        @test typeof.(soundspeed_air.(ts)) == typeof.(e_int)
        @test typeof.(gas_constant_air.(ts)) == typeof.(e_int)
        @test typeof.(specific_enthalpy.(ts)) == typeof.(e_int)
        @test typeof.(vapor_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(relative_humidity.(ts)) == typeof.(e_int)
        @test typeof.(air_pressure.(ts)) == typeof.(e_int)
        @test typeof.(air_density.(ts)) == typeof.(e_int)
        @test typeof.(total_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(liquid_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(ice_specific_humidity.(ts)) == typeof.(e_int)
        @test typeof.(cp_m.(ts)) == typeof.(e_int)
        @test typeof.(cv_m.(ts)) == typeof.(e_int)
        @test typeof.(air_temperature.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_sat.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_dry.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_vapor.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_liquid.(ts)) == typeof.(e_int)
        @test typeof.(internal_energy_ice.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_vapor.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_sublim.(ts)) == typeof.(e_int)
        @test typeof.(latent_heat_fusion.(ts)) == typeof.(e_int)
        @test typeof.(q_vap_saturation.(ts)) == typeof.(e_int)
        @test typeof.(q_vap_saturation_liquid.(ts)) == typeof.(e_int)
        @test typeof.(q_vap_saturation_ice.(ts)) == typeof.(e_int)
        @test typeof.(saturation_excess.(ts)) == typeof.(e_int)
        @test typeof.(liquid_fraction.(ts)) == typeof.(e_int)
        @test typeof.(liquid_ice_pottemp.(ts)) == typeof.(e_int)
        @test typeof.(dry_pottemp.(ts)) == typeof.(e_int)
        @test typeof.(exner.(ts)) == typeof.(e_int)
        @test typeof.(liquid_ice_pottemp_sat.(ts)) == typeof.(e_int)
        @test typeof.(specific_volume.(ts)) == typeof.(e_int)
        @test typeof.(supersaturation.(ts, Ice())) == typeof.(e_int)
        @test typeof.(supersaturation.(ts, Liquid())) == typeof.(e_int)
        @test typeof.(virtual_pottemp.(ts)) == typeof.(e_int)
        @test eltype.(gas_constants.(ts)) == typeof.(e_int)

        @test typeof.(total_specific_enthalpy.(ts, e_tot)) == typeof.(e_int)
        @test typeof.(moist_static_energy.(ts, e_pot)) == typeof.(e_int)
        @test typeof.(getproperty.(PhasePartition.(ts), :tot)) == typeof.(e_int)
    end

end

@testset "Thermodynamics - dry limit" begin

    ArrayType = Array{Float64}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack T, p, RS, e_int, ρ, θ_liq_ice, phase_type = profiles
    @unpack q_tot, q_liq, q_ice, q_pt, RH, e_kin, e_pot = profiles

    # PhasePartition test is noisy, so do this only once:
    ts_dry = PhaseDry(param_set, first(e_int), first(ρ))
    ts_eq = PhaseEquil(param_set, first(e_int), first(ρ), typeof(first(ρ))(0))
    @test PhasePartition(ts_eq).tot ≈ PhasePartition(ts_dry).tot
    @test PhasePartition(ts_eq).liq ≈ PhasePartition(ts_dry).liq
    @test PhasePartition(ts_eq).ice ≈ PhasePartition(ts_dry).ice

    @test mixing_ratios(ts_eq).tot ≈ mixing_ratios(ts_dry).tot
    @test mixing_ratios(ts_eq).liq ≈ mixing_ratios(ts_dry).liq
    @test mixing_ratios(ts_eq).ice ≈ mixing_ratios(ts_dry).ice

    ts_dry = PhaseDry.(param_set, e_int, ρ)
    ts_eq = PhaseEquil.(param_set, e_int, ρ, q_tot .* 0)

    @test all(gas_constant_air.(ts_eq) .≈ gas_constant_air.(ts_dry))
    @test all(relative_humidity.(ts_eq) .≈ relative_humidity.(ts_dry))
    @test all(air_pressure.(ts_eq) .≈ air_pressure.(ts_dry))
    @test all(air_density.(ts_eq) .≈ air_density.(ts_dry))
    @test all(specific_volume.(ts_eq) .≈ specific_volume.(ts_dry))
    @test all(
        total_specific_humidity.(ts_eq) .≈ total_specific_humidity.(ts_dry),
    )
    @test all(
        liquid_specific_humidity.(ts_eq) .≈ liquid_specific_humidity.(ts_dry),
    )
    @test all(ice_specific_humidity.(ts_eq) .≈ ice_specific_humidity.(ts_dry))
    @test all(cp_m.(ts_eq) .≈ cp_m.(ts_dry))
    @test all(cv_m.(ts_eq) .≈ cv_m.(ts_dry))
    @test all(air_temperature.(ts_eq) .≈ air_temperature.(ts_dry))
    @test all(internal_energy.(ts_eq) .≈ internal_energy.(ts_dry))
    @test all(internal_energy_sat.(ts_eq) .≈ internal_energy_sat.(ts_dry))
    @test all(internal_energy_dry.(ts_eq) .≈ internal_energy_dry.(ts_dry))
    @test all(internal_energy_vapor.(ts_eq) .≈ internal_energy_vapor.(ts_dry))
    @test all(internal_energy_liquid.(ts_eq) .≈ internal_energy_liquid.(ts_dry))
    @test all(internal_energy_ice.(ts_eq) .≈ internal_energy_ice.(ts_dry))
    @test all(soundspeed_air.(ts_eq) .≈ soundspeed_air.(ts_dry))
    @test all(supersaturation.(ts_eq, Ice()) .≈ supersaturation.(ts_dry, Ice()))
    @test all(
        supersaturation.(ts_eq, Liquid()) .≈ supersaturation.(ts_dry, Liquid()),
    )
    @test all(latent_heat_vapor.(ts_eq) .≈ latent_heat_vapor.(ts_dry))
    @test all(latent_heat_sublim.(ts_eq) .≈ latent_heat_sublim.(ts_dry))
    @test all(latent_heat_fusion.(ts_eq) .≈ latent_heat_fusion.(ts_dry))
    @test all(q_vap_saturation.(ts_eq) .≈ q_vap_saturation.(ts_dry))
    @test all(
        q_vap_saturation_liquid.(ts_eq) .≈ q_vap_saturation_liquid.(ts_dry),
    )
    @test all(q_vap_saturation_ice.(ts_eq) .≈ q_vap_saturation_ice.(ts_dry))
    @test all(saturation_excess.(ts_eq) .≈ saturation_excess.(ts_dry))
    @test all(liquid_fraction.(ts_eq) .≈ liquid_fraction.(ts_dry))
    @test all(liquid_ice_pottemp.(ts_eq) .≈ liquid_ice_pottemp.(ts_dry))
    @test all(dry_pottemp.(ts_eq) .≈ dry_pottemp.(ts_dry))
    @test all(virtual_pottemp.(ts_eq) .≈ virtual_pottemp.(ts_dry))
    @test all(liquid_ice_pottemp_sat.(ts_eq) .≈ liquid_ice_pottemp_sat.(ts_dry))
    @test all(exner.(ts_eq) .≈ exner.(ts_dry))

    @test all(
        saturation_vapor_pressure.(ts_eq, Ice()) .≈
        saturation_vapor_pressure.(ts_dry, Ice()),
    )
    @test all(
        saturation_vapor_pressure.(ts_eq, Liquid()) .≈
        saturation_vapor_pressure.(ts_dry, Liquid()),
    )
    @test all(first.(gas_constants.(ts_eq)) ≈ first.(gas_constants.(ts_dry)))
    @test all(last.(gas_constants.(ts_eq)) ≈ last.(gas_constants.(ts_dry)))

end

@testset "Thermodynamics - ProfileSet Iterator" begin
    ArrayType = Array{Float64}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)
    @unpack T, q_pt, z, phase_type = profiles
    @test all(z .≈ (nt.z for nt in profiles))
    @test all(T .≈ (nt.T for nt in profiles))
    @test all(getproperty.(q_pt, :tot) .≈ (nt.q_pt.tot for nt in profiles))
    @test all(phase_type .== (nt.phase_type for nt in profiles))
end

@testset "Thermodynamics - Performance" begin
    ArrayType = Array{Float64}
    FT = eltype(ArrayType)
    profiles = PhaseEquilProfiles(param_set, ArrayType)

    @unpack e_int, ρ, q_tot = profiles

    @btime TD.PhaseEquil_dev_only.(
        $param_set,
        $e_int,
        $ρ,
        $q_tot;
        sat_adjust_method = NewtonsMethod,
    )

    @btime TD.PhaseEquil_dev_only.(
        $param_set,
        $e_int,
        $ρ,
        $q_tot;
        sat_adjust_method = RegulaFalsiMethod,
        maxiter = 20,
    )

    # Fails to converge:
    # @btime TD.PhaseEquil_dev_only.(
    #     $param_set,
    #     $e_int,
    #     $ρ,
    #     $q_tot;
    #     sat_adjust_method = NewtonsMethodAD,
    #     maxiter = 50,
    # )

    @btime TD.PhaseEquil_dev_only.(
        $param_set,
        $e_int,
        $ρ,
        $q_tot;
        sat_adjust_method = SecantMethod,
        maxiter = 50,
    )

end

end # module
