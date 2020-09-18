using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0
using CLIMAParameters.Atmos.Microphysics: K_therm
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Land

@testset "Land heat parameterizations" begin
    FT = Float64
    # Density of liquid water (kg/m``^3``)
    _ρ_l = FT(ρ_cloud_liq(param_set))
    # Density of ice water (kg/m``^3``)
    _ρ_i = FT(ρ_cloud_ice(param_set))
    # Volum. isobaric heat capacity liquid water (J/m3/K)
    _ρcp_l = FT(cp_l(param_set) * _ρ_l)
    # Volumetric isobaric heat capacity ice (J/m3/K)
    _ρcp_i = FT(cp_i(param_set) * _ρ_i)
    # Reference temperature (K)
    _T_ref = FT(T_0(param_set))
    # Latent heat of fusion at ``T_0`` (J/kg)
    _LH_f0 = FT(LH_f0(param_set))
    # Thermal conductivity of dry air
    κ_air = FT(K_therm(param_set))

    @test temperature_from_ρe_int(5.4e7, 0.05, 2.1415e6, param_set) ==
          FT(_T_ref + (5.4e7 + 0.05 * _ρ_i * _LH_f0) / 2.1415e6)

    @test volumetric_heat_capacity(0.25, 0.05, 1e6, param_set) ==
          FT(1e6 + 0.25 * _ρcp_l + 0.05 * _ρcp_i)

    @test volumetric_internal_energy(0.05, 2.1415e6, 300.0, param_set) ==
          FT(2.1415e6 * (300.0 - _T_ref) - 0.05 * _ρ_i * _LH_f0)

    @test saturated_thermal_conductivity(0.25, 0.05, 0.57, 2.29) ==
          FT(0.57^(0.25 / (0.05 + 0.25)) * 2.29^(0.05 / (0.05 + 0.25)))

    @test saturated_thermal_conductivity(0.0, 0.0, 0.57, 2.29) == FT(0.0)

    @test relative_saturation(0.25, 0.05, 0.4) == FT((0.25 + 0.05) / 0.4)

    # Test branching in kersten_number
    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.2,
        ν_ss_gravel = 0.1,
        ν_ss_om = 0.1,
        ν_ss_quartz = 0.1,
        κ_solid = 0.1,
        ρp = 1,
    )
    # ice fraction = 0
    @test kersten_number(0.0, 0.75, soil_param_functions) == FT(
        0.75^((FT(1) + 0.1 - 0.24 * 0.1 - 0.1) / FT(2)) *
        (
            (FT(1) + exp(-18.1 * 0.75))^(-FT(3)) -
            ((FT(1) - 0.75) / FT(2))^FT(3)
        )^(FT(1) - 0.1),
    )

    # ice fraction ~= 0
    @test kersten_number(0.05, 0.75, soil_param_functions) ==
          FT(0.75^(FT(1) + 0.1))

    @test thermal_conductivity(1.5, 0.7287, 0.7187) ==
          FT(0.7287 * 0.7187 + (FT(1) - 0.7287) * 1.5)

    @test volumetric_internal_energy_liq(300.0, param_set) ==
          FT(_ρcp_l * (300.0 - _T_ref))

    @test k_solid(FT(0.5), FT(0.25), FT(2.0), FT(3.0), FT(2.0)) ==
          FT(2)^FT(0.5) * FT(2)^FT(0.25) * FT(3.0)^FT(0.25)

    @test ksat_frozen(FT(0.5), FT(0.1), FT(0.4)) ==
          FT(0.5)^FT(0.9) * FT(0.4)^FT(0.1)

    @test ksat_unfrozen(FT(0.5), FT(0.1), FT(0.4)) ==
          FT(0.5)^FT(0.9) * FT(0.4)^FT(0.1)

    @test k_dry(param_set, soil_param_functions) ==
          ((FT(0.053) * FT(0.1) - κ_air) * FT(0.8) + κ_air * FT(1.0)) /
          (FT(1.0) - (FT(1.0) - FT(0.053)) * FT(0.8))
end
