#!/usr/bin/env julia --project

using ArgParse
using LinearAlgebra
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.TurbulenceClosures
using ClimateMachine.SystemSolvers: ManyColumnLU
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics:
    air_density, air_temperature, total_energy, internal_energy, PhasePartition, air_pressure
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using CLIMAParameters
using CLIMAParameters.Planet:
    day, grav, R_d, Omega, planet_radius, MSLP, T_triple, press_triple, LH_v0, R_v
using CLIMAParameters.Atmos.SubgridScale:
    C_drag

struct EarthParameterSet <: AbstractEarthParameterSet end
#press_triple(::EarthParameterSet) = 610.78      # from Thatcher and Jablonowski (2016)
#C_drag(::EarthParameterSet) = 0.0044            # "

# driver-specific parameters added here
T_sfc_pole(::EarthParameterSet) = 271.0

const param_set = EarthParameterSet()


"""
Defines analytical function for prescribed T_sfc and q_sfc, following
Thatcher and Jablonowski (2016), used to calculate bulk surface fluxes.

T_sfc_pole = SST at the poles (default: 271 K), specified above
"""
struct Varying_SST_TJ16{PS, O, MM}
    param_set::PS
    orientation::O
    moisture::MM
end
function (st::Varying_SST_TJ16)(state, aux, t)
    FT = eltype(state)
    φ = latitude(st.orientation, aux)

    _T_sfc_pole = T_sfc_pole(st.param_set)

    Δφ = FT(26) * π / FT(180)   # latitudinal width of Gaussian function
    ΔSST = FT(29)               # Eq-pole SST difference in K
    T_sfc = ΔSST * exp(-φ^2 / (2 * Δφ^2)) + _T_sfc_pole

    eps =  FT(0.622)
    ρ = state.ρ

    q_tot = state.moisture.ρq_tot / ρ
    q = PhasePartition(q_tot) 

    e_int = internal_energy( st.moisture, st.orientation, state, aux)
    T = air_temperature( st.param_set, e_int, q)
    p = air_pressure( st.param_set, T, ρ, q)

    _T_triple = T_triple(st.param_set)          # triple point of water
    _press_triple = press_triple(st.param_set)  # sat water pressure at T_triple
    _LH_v0 = LH_v0(st.param_set)                # latent heat of vaporization at T_triple
    _R_v = R_v(st.param_set)                    # gas constant for water vapor

    q_sfc = eps / p * _press_triple * exp(-_LH_v0 / _R_v * (FT(1) / T_sfc - FT(1) / _T_triple))

    return T_sfc, q_sfc
end
