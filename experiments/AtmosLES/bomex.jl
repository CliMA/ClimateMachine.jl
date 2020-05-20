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
# This setup works in both Float32 and Float64 precision. `FT`
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

@article{doi:10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2,
author = {Siebesma, A. Pier and Bretherton,
          Christopher S. and Brown,
          Andrew and Chlond,
          Andreas and Cuxart,
          Joan and Duynkerke,
          Peter G. and Jiang,
          Hongli and Khairoutdinov,
          Marat and Lewellen,
          David and Moeng,
          Chin-Hoh and Sanchez,
          Enrique and Stevens,
          Bjorn and Stevens,
          David E.},
title = {A Large Eddy Simulation Intercomparison Study of Shallow Cumulus Convection},
journal = {Journal of the Atmospheric Sciences},
volume = {60},
number = {10},
pages = {1201-1219},
year = {2003},
doi = {10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2},
URL = {https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2},
eprint = {https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2}
=#

using ClimateMachine
ClimateMachine.init()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.ODESolvers
using ClimateMachine.MoistThermodynamics
using ClimateMachine.VariableTemplates

using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.DGmethods: vars_state_conservative, vars_state_auxiliary
import ClimateMachine.Atmos: source!, atmos_source!, altitude
import ClimateMachine.Atmos: flux_second_order!, thermo_state

"""
  Bomex Geostrophic Forcing (Source)
"""
struct BomexGeostrophic{FT} <: Source
    "Coriolis parameter [s⁻¹]"
    f_coriolis::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end
function atmos_source!(
    s::BomexGeostrophic,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)

    f_coriolis = s.f_coriolis
    u_geostrophic = s.u_geostrophic
    u_slope = s.u_slope
    v_geostrophic = s.v_geostrophic

    z = altitude(atmos, aux)
    # Note z dependence of eastward geostrophic velocity
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(atmos, aux)
    fkvector = f_coriolis * ẑ
    # Accumulate sources
    source.ρu -= fkvector × (state.ρu .- state.ρ * u_geo)
    return nothing
end

"""
  Bomex Sponge (Source)
"""
struct BomexSponge{FT} <: Source
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
function atmos_source!(
    s::BomexSponge,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)

    z_max = s.z_max
    z_sponge = s.z_sponge
    α_max = s.α_max
    γ = s.γ
    u_geostrophic = s.u_geostrophic
    u_slope = s.u_slope
    v_geostrophic = s.v_geostrophic

    z = altitude(atmos, aux)
    u_geo = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
    ẑ = vertical_unit_vector(atmos, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α_max * sinpi(r / 2)^s.γ
        source.ρu -= β_sponge * (state.ρu .- state.ρ * u_geo)
    end
    return nothing
end

"""
  BomexTendencies (Source)
Moisture, Temperature and Subsidence tendencies
"""
struct BomexTendencies{FT} <: Source
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
function atmos_source!(
    s::BomexTendencies,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)
    ρ = state.ρ
    z = altitude(atmos, aux)
    _e_int_v0 = FT(e_int_v0(atmos.param_set))

    # Establish thermodynamic state
    TS = thermo_state(atmos, state, aux)

    # Moisture tendencey (sink term)
    # Temperature tendency (Radiative cooling)
    # Large scale subsidence
    # Unpack struct
    zl_moisture = s.zl_moisture
    zh_moisture = s.zh_moisture
    z_max = s.z_max
    zl_sub = s.zl_sub
    zh_sub = s.zh_sub
    zl_temperature = zl_sub
    w_sub = s.w_sub
    ∂qt∂t_peak = s.∂qt∂t_peak
    ∂θ∂t_peak = s.∂θ∂t_peak
    k̂ = vertical_unit_vector(atmos, aux)

    # Thermodynamic state identification
    q_pt = PhasePartition(TS)
    cvm = cv_m(TS)

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

    # Collect Sources
    source.moisture.ρq_tot += ρ∂qt∂t
    source.ρe += cvm * ρ∂θ∂t * exner(TS) + _e_int_v0 * ρ∂qt∂t
    source.ρe -= ρ * w_s * dot(k̂, diffusive.∇h_tot)
    source.moisture.ρq_tot -= ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
    return nothing
end

"""
  Initial Condition for BOMEX LES
"""
seed = MersenneTwister(0)
function init_bomex!(bl, state, aux, (x, y, z), t)
    # This experiment runs the BOMEX LES Configuration
    # (Shallow cumulus cloud regime)
    # x,y,z imply eastward, northward and altitude coordinates in `[m]`

    # Problem floating point precision
    FT = eltype(state)

    P_sfc::FT = 1.015e5 # Surface air pressure
    qg::FT = 22.45e-3 # Total moisture at surface
    q_pt_sfc = PhasePartition(qg) # Surface moisture partitioning
    Rm_sfc = gas_constant_air(bl.param_set, q_pt_sfc) # Moist gas constant
    θ_liq_sfc = FT(299.1) # Prescribed θ_liq at surface
    T_sfc = FT(300.4) # Surface temperature
    _grav = FT(grav(bl.param_set))

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
    TS = LiquidIcePotTempSHumEquil_given_pressure(bl.param_set, θ_liq, P, q_tot)
    T = air_temperature(TS)
    ρ = air_density(TS)
    q_pt = PhasePartition(TS)

    # Compute momentum contributions
    ρu = ρ * u
    ρv = ρ * v
    ρw = ρ * w

    # Compute energy contributions
    e_kin = FT(1 // 2) * (u^2 + v^2 + w^2)
    e_pot = _grav * z
    ρe_tot = ρ * total_energy(e_kin, e_pot, TS)

    # Assign initial conditions for prognostic state variables
    state.ρ = ρ
    state.ρu = SVector(ρu, ρv, ρw)
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_tot

    if z <= FT(400) # Add random perturbations to bottom 400m of model
        state.ρe += rand(seed) * ρe_tot / 100
        state.moisture.ρq_tot += rand(seed) * ρ * q_tot / 100
    end
end

function config_bomex(FT, N, resolution, xmax, ymax, zmax)

    ics = init_bomex!     # Initial conditions

    C_smag = FT(0.18)     # Smagorinsky coefficient

    u_star = FT(0.28)     # Friction velocity

    T_sfc = FT(300.4)     # Surface temperature `[K]`
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
    source = (
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
    )

    # Choose multi-rate explicit solver
    ode_solver_type = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    # Assemble model components
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly{FT}(C_smag),
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(0.1)),
        source = source,
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    # normPu_int is the internal horizontal speed
                    # P represents the projection onto the horizontal
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) -> moisture_flux,
                ),
            ),
            AtmosBC(),
        ),
        init_state_conservative = ics,
    )

    # Assemble configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "BOMEX",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_bomex!,
        solver_type = ode_solver_type,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    default_dgngrp =
        setup_atmos_default_diagnostics("2500steps", driver_config.name)
    core_dgngrp = setup_atmos_core_diagnostics("2500steps", driver_config.name)
    return ClimateMachine.DiagnosticsConfiguration([
        default_dgngrp,
        core_dgngrp,
    ])
end

function main()
    FT = Float32

    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(100)
    Δv = FT(40)

    resolution = (Δh, Δh, Δv)

    # Prescribe domain parameters
    xmax = FT(6400)
    ymax = FT(6400)
    zmax = FT(3000)

    t0 = FT(0)

    # For a full-run, please set the timeend to 3600*6 seconds
    # For the test we set this to == 30 minutes
    timeend = FT(1800)
    #timeend = FT(3600 * 6)
    CFLmax = FT(8)

    driver_config = config_bomex(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 4e-3)
end

main()
