#!/usr/bin/env julia --project
#=
# This experiment file establishes the initial conditions, boundary conditions,
# source terms and simulation parameters (domain size + resolution) for the
# BOMEX LES case. The set of parameters presented in the `master` branch copy
# include those that have passed offline tests at the full simulation time of
# 6 hours. Suggested offline tests included plotting horizontal-domain averages
# of key properties (see AtmosDiagnostics). The timestepper configuration is in
# `src/Driver/solver_configs.jl` while the `AtmosModel` defaults can be found in
# `src/Atmos/Model/AtmosModel.jl` and `src/Driver/driver_configs.jl`
#
# To simulate the full 6 hour experiment, change `timeend` to (3600*6) and type in
#
# julia --project experiments/AtmosLES/bomex.jl
#
# See `src/Driver/driver_configs.jl` for additional flags (e.g. VTK, diagnostics,
# update-interval, output directory settings)
#
# Upcoming changes:
# 1) Atomic sources
# 2) Improved boundary conditions
# 3) Collapsed experiment design
# 4) Updates to generally keep this in sync with master

[Siebesma2003](@cite)
=#

using ArgParse
using Distributions
using DocStringExtensions
using LinearAlgebra
using Printf
using StaticArrays
using Test

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.TurbulenceConvection
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws: prognostic_vars

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day
using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.BalanceLaws: source, prognostic_vars
using ClimateMachine.Atmos: altitude, recover_thermo_state
using UnPack

"""
  Bomex Geostrophic Forcing (Source)
"""
struct BomexGeostrophic{FT} <: TendencyDef{Source}
    "Coriolis parameter [s⁻¹]"
    f_coriolis::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end

prognostic_vars(::BomexGeostrophic) = (Momentum(),)

function source(::Momentum, s::BomexGeostrophic, m, args)
    @unpack state, aux = args
    @unpack f_coriolis, u_geostrophic, u_slope, v_geostrophic = s

    z = altitude(m, aux)
    # Note z dependence of eastward geostrophic velocity
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    fkvector = f_coriolis * ẑ
    # Accumulate sources
    return -fkvector × (state.ρu .- state.ρ * u_geo)
end

"""
  Bomex Sponge (Source)
"""
struct BomexSponge{FT} <: TendencyDef{Source}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end

prognostic_vars(::BomexSponge) = (Momentum(),)

function source(::Momentum, s::BomexSponge, m, args)
    @unpack state, aux = args
    @unpack z_max, z_sponge, α_max, γ = s
    @unpack u_geostrophic, u_slope, v_geostrophic = s

    z = altitude(m, aux)
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(m, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α_max * sinpi(r / 2)^s.γ
        return -β_sponge * (state.ρu .- state.ρ * u_geo)
    else
        FT = eltype(state)
        return SVector{3, FT}(0, 0, 0)
    end
end

"""
    BomexTendencies (Source)

Moisture, Temperature and Subsidence tendencies
"""
struct BomexTendencies{FT} <: TendencyDef{Source}
    "Advection tendency in total moisture `[s⁻¹]`"
    ∂qt∂t_peak::FT
    "Lower extent of piecewise profile (moisture term) `[m]`"
    zl_moisture::FT
    "Upper extent of piecewise profile (moisture term) `[m]`"
    zh_moisture::FT
    "Cooling rate `[K/s]`"
    ∂θ∂t_peak::FT
    "Lower extent of piecewise profile (subsidence term) `[m]`"
    zl_sub::FT
    "Upper extent of piecewise profile (subsidence term) `[m]`"
    zh_sub::FT
    "Subsidence peak velocity"
    w_sub::FT
    "Max height in domain"
    z_max::FT
end

prognostic_vars(::BomexTendencies) = (Mass(), Energy(), TotalMoisture())

function compute_bomex_tend_params(s, m, args)
    @unpack state, aux = args
    FT = eltype(state)
    ρ = state.ρ
    z = altitude(m, aux)

    # Moisture tendencey (sink term)
    # Temperature tendency (Radiative cooling)
    # Large scale subsidence
    # Unpack struct
    @unpack zl_moisture, zh_moisture, z_max, zl_sub, zh_sub = s
    @unpack w_sub, ∂qt∂t_peak, ∂θ∂t_peak = s
    zl_temperature = zl_sub
    k̂ = vertical_unit_vector(m, aux)

    # Piecewise term for moisture tendency
    linscale_moisture = (z - zl_moisture) / (zh_moisture - zl_moisture)
    if z <= zl_moisture
        ρ∂qt∂t = ρ * ∂qt∂t_peak
    elseif zl_moisture < z <= zh_moisture
        ρ∂qt∂t = ρ * (∂qt∂t_peak - ∂qt∂t_peak * linscale_moisture)
    else
        ρ∂qt∂t = -zero(FT)
    end

    # Piecewise term for internal energy tendency
    linscale_temp = (z - zl_sub) / (z_max - zl_sub)
    if z <= zl_sub
        ρ∂θ∂t = ρ * ∂θ∂t_peak
    elseif zl_temperature < z <= z_max
        ρ∂θ∂t = ρ * (∂θ∂t_peak - ∂θ∂t_peak * linscale_temp)
    else
        ρ∂θ∂t = -zero(FT)
    end

    # Piecewise terms for subsidence
    linscale_sub = (z - zl_sub) / (zh_sub - zl_sub)
    w_s = -zero(FT)
    if z <= zl_sub
        w_s = -zero(FT) + z * (w_sub) / (zl_sub)
    elseif zl_sub < z <= zh_sub
        w_s = w_sub - (w_sub) * linscale_sub
    else
        w_s = -zero(FT)
    end
    return (w_s = w_s, ρ∂qt∂t = ρ∂qt∂t, ρ∂θ∂t = ρ∂θ∂t, k̂ = k̂)
end

function source(::Mass, s::BomexTendencies, m, args)
    @unpack state, diffusive = args
    params = compute_bomex_tend_params(s, m, args)
    @unpack ρ∂qt∂t, w_s, k̂ = params
    ρ = state.ρ
    return ρ∂qt∂t - state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end
function source(::Energy, s::BomexTendencies, m, args)
    @unpack state, diffusive = args
    @unpack ts = args.precomputed
    params = compute_bomex_tend_params(s, m, args)
    FT = eltype(state)
    @unpack ρ∂qt∂t, ρ∂θ∂t, w_s, k̂ = params
    cvm = cv_m(ts)
    param_set = parameter_set(m)
    _e_int_v0 = FT(e_int_v0(param_set))
    term1 = cvm * ρ∂θ∂t * exner(ts) + _e_int_v0 * ρ∂qt∂t
    term2 = state.ρ * w_s * dot(k̂, diffusive.energy.∇h_tot)
    return term1 - term2
end
function source(::TotalMoisture, s::BomexTendencies, m, args)
    @unpack state, diffusive = args
    params = compute_bomex_tend_params(s, m, args)
    @unpack ρ∂qt∂t, w_s, k̂ = params
    return ρ∂qt∂t - state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end

add_perturbations!(state, localgeo) = nothing

"""
  Initial Condition for BOMEX LES
"""
function init_bomex!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    # This experiment runs the BOMEX LES Configuration
    # (Shallow cumulus cloud regime)
    # x,y,z imply eastward, northward and altitude coordinates in `[m]`

    # Problem floating point precision
    FT = eltype(state)
    param_set = parameter_set(bl)

    P_sfc::FT = 1.015e5 # Surface air pressure
    qg::FT = 22.45e-3 # Total moisture at surface
    q_pt_sfc = PhasePartition(qg) # Surface moisture partitioning
    Rm_sfc = gas_constant_air(param_set, q_pt_sfc) # Moist gas constant
    θ_liq_sfc = FT(299.1) # Prescribed θ_liq at surface
    T_sfc = FT(300.4) # Surface temperature
    _grav = FT(grav(param_set))

    # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
    u::FT = 0
    v::FT = 0
    w::FT = 0

    # Prescribed altitudes for piece-wise profile construction
    zl1::FT = 520
    zl2::FT = 1480
    zl3::FT = 2000
    zl4::FT = 3000

    # Assign piecewise quantities to θ_liq and q_tot
    θ_liq::FT = 0
    q_tot::FT = 0

    # Piecewise functions for potential temperature and total moisture
    if FT(0) <= z <= zl1
        # Well mixed layer
        θ_liq = 298.7
        q_tot = 17.0 + (z / zl1) * (16.3 - 17.0)
    elseif z > zl1 && z <= zl2
        # Conditionally unstable layer
        θ_liq = 298.7 + (z - zl1) * (302.4 - 298.7) / (zl2 - zl1)
        q_tot = 16.3 + (z - zl1) * (10.7 - 16.3) / (zl2 - zl1)
    elseif z > zl2 && z <= zl3
        # Absolutely stable inversion
        θ_liq = 302.4 + (z - zl2) * (308.2 - 302.4) / (zl3 - zl2)
        q_tot = 10.7 + (z - zl2) * (4.2 - 10.7) / (zl3 - zl2)
    else
        θ_liq = 308.2 + (z - zl3) * (311.85 - 308.2) / (zl4 - zl3)
        q_tot = 4.2 + (z - zl3) * (3.0 - 4.2) / (zl4 - zl3)
    end

    # Set velocity profiles - piecewise profile for u
    zlv::FT = 700
    if z <= zlv
        u = -8.75
    else
        u = -8.75 + (z - zlv) * (-4.61 + 8.75) / (zl4 - zlv)
    end

    # Convert total specific humidity to kg/kg
    q_tot /= 1000
    # Scale height based on surface parameters
    H = Rm_sfc * T_sfc / _grav
    # Pressure based on scale height
    P = P_sfc * exp(-z / H)

    # Establish thermodynamic state and moist phase partitioning
    ts = PhaseEquil_pθq(param_set, P, θ_liq, q_tot)
    T = air_temperature(ts)
    ρ = air_density(ts)

    # Compute momentum contributions
    ρu = ρ * u
    ρv = ρ * v
    ρw = ρ * w

    # Compute energy contributions
    e_kin = FT(1 // 2) * (u^2 + v^2 + w^2)
    e_pot = _grav * z
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)

    # Assign initial conditions for prognostic state variables
    state.ρ = ρ
    state.ρu = SVector(ρu, ρv, ρw)
    state.energy.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_tot
    if moisture_model(bl) isa NonEquilMoist
        state.moisture.ρq_liq = FT(0)
        state.moisture.ρq_ice = FT(0)
    end

    add_perturbations!(state, localgeo)
    init_state_prognostic!(turbconv_model(bl), bl, state, aux, localgeo, t)
end

function bomex_model(
    ::Type{FT},
    config_type,
    zmax,
    surface_flux;
    turbconv = NoTurbConv(),
    moisture_model = "equilibrium",
) where {FT}

    ics = init_bomex!     # Initial conditions

    C_smag = FT(0.23)     # Smagorinsky coefficient

    u_star = FT(0.28)     # Friction velocity
    C_drag = FT(0.0011)   # Bulk transfer coefficient

    T_sfc = FT(300.4)     # Surface temperature `[K]`
    q_sfc = FT(22.45e-3)  # Surface specific humiity `[kg/kg]`
    LHF = FT(147.2)       # Latent heat flux `[W/m²]`
    SHF = FT(9.5)         # Sensible heat flux `[W/m²]`
    moisture_flux = LHF / latent_heat_vapor(param_set, T_sfc)

    ∂qt∂t_peak = FT(-1.2e-8)  # Moisture tendency (energy source)
    zl_moisture = FT(300)     # Low altitude limit for piecewise function (moisture source)
    zh_moisture = FT(500)     # High altitude limit for piecewise function (moisture source)
    ∂θ∂t_peak = FT(-2 / FT(day(param_set)))  # Potential temperature tendency (energy source)

    z_sponge = FT(2400)     # Start of sponge layer
    α_max = FT(0.75)        # Strength of sponge layer (timescale)
    γ = 2              # Strength of sponge layer (exponent)

    u_geostrophic = FT(-10)        # Eastward relaxation speed
    u_slope = FT(1.8e-3)     # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)          # Northward relaxation speed

    zl_sub = FT(1500)         # Low altitude for piecewise function (subsidence source)
    zh_sub = FT(2100)         # High altitude for piecewise function (subsidence source)
    w_sub = FT(-0.65e-2)     # Subsidence velocity peak value

    f_coriolis = FT(0.376e-4) # Coriolis parameter

    # Assemble source components
    source_default = (
        Gravity(),
        BomexTendencies{FT}(
            ∂qt∂t_peak,
            zl_moisture,
            zh_moisture,
            ∂θ∂t_peak,
            zl_sub,
            zh_sub,
            w_sub,
            zmax,
        ),
        BomexSponge{FT}(
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        BomexGeostrophic{FT}(f_coriolis, u_geostrophic, u_slope, v_geostrophic),
        turbconv_sources(turbconv)...,
    )
    if moisture_model == "equilibrium"
        source = source_default
        moisture = EquilMoist(; maxiter = 5, tolerance = FT(0.1))
    elseif moisture_model == "nonequilibrium"
        source = (source_default..., CreateClouds())
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model, using the defaults""",
            moisture_model,
        )
        source = source_default
        moisture = EquilMoist(; maxiter = 5, tolerance = FT(0.1))
    end

    # Set up problem initial and boundary conditions
    if surface_flux == "prescribed"
        energy_bc = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF)
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> moisture_flux)
    elseif surface_flux == "bulk"
        energy_bc = BulkFormulaEnergy(
            (bl, state, aux, t, normPu_int) -> C_drag,
            (bl, state, aux, t) -> (T_sfc, q_sfc),
        )
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag,
            (state, aux, t) -> q_sfc,
        )
    else
        @warn @sprintf(
            """
%s: unrecognized surface flux; using 'prescribed'""",
            surface_flux,
        )
    end

    # Assemble model components
    physics = AtmosPhysics{FT}(
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        moisture = moisture,
        turbconv = turbconv,
    )

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                physics;
                momentum = Impenetrable(DragLaw(
                    # normPu_int is the internal horizontal speed
                    # P represents the projection onto the horizontal
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = energy_bc,
                moisture = moisture_bc,
                turbconv = turbconv_bcs(turbconv)[1],
            ),
            AtmosBC(physics; turbconv = turbconv_bcs(turbconv)[2]),
        ),
        init_state_prognostic = ics,
    )

    # Assemble model components
    model =
        AtmosModel{FT}(config_type, physics; problem = problem, source = source)

    return model
end
