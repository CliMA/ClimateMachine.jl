using MPI
using OrderedCollections
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, ρ_cloud_ice, cp_l, cp_i, T_0, LH_f0
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
        ν_gravel = 0.1,
        ν_om = 0.1,
        ν_sand = 0.1,
        a = 0.24,
        b = 18.1,
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
end
