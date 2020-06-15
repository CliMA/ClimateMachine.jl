#!/usr/bin/env julia --project
#=
# This experiment file establishes the initial conditions, boundary conditions,
# source terms and simulation parameters (domain size + resolution) for the
# ARM LES case. The set of parameters presented in the `master` branch copy
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
# julia --project experiments/AtmosLES/arm.jl
#
# See `src/Driver/driver_configs.jl` for additional flags (e.g. VTK, diagnostics,
# update-interval, output directory settings)
#
# Upcoming changes:
# 3) Collapsed experiment design

=#

using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates

using Distributions
using DocStringExtensions
using Dierckx
using LinearAlgebra
using Random
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.DGMethods: vars_state_conservative, vars_state_auxiliary
import ClimateMachine.Atmos: source!, atmos_source!, altitude
import ClimateMachine.Atmos: flux_second_order!, thermo_state

"""
  ARM Geostrophic Forcing (Source)
"""
struct ARMGeostrophic{FT} <: Source
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
    s::ARMGeostrophic,
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
  ARM Sponge (Source)
"""
struct ARMSponge{FT} <: Source
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
    s::ARMSponge,
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
  ARMTendencies (Source)
Moisture, Temperature and Subsidence tendencies
"""
struct ARMTendencies{FT} <: Source
end
function atmos_source!(
    s::ARMTendencies,
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

    w_sub = FT(0) # TODO fix value correctly
    k̂ = vertical_unit_vector(atmos, aux)

    # Thermodynamic state identification
    q_pt = PhasePartition(TS)
    cvm = cv_m(TS)

    # ARM Tendencies Depend on time, but not on the spatial coordinate
    hours = 3600

    if t <= FT(3hours)
        ∂θ∂t_adv = FT(0)
        ∂θ∂t_rad = FT(-0.125) + (t - 3hours) * FT(0.125) / FT(3hours)
        ∂qt∂t = 0.08 + (t/3hours) * FT(0.02 - 0.08)
    elseif t > FT(3hours) && t <= FT(6hours)
        ∂θ∂t_adv = FT(0) 
        ∂θ∂t_rad = FT(0)
        ∂qt∂t = 0.02 + (t - 3hours) * FT(0.04 - 0.02) / FT(3hours)
    elseif t > FT(6hours) && z <= FT(9hours)
        ∂θ∂t_adv = FT(0) + (t - 6hours) * FT(-0.08) / FT(3hours)
        ∂θ∂t_rad = FT(0)
        ∂qt∂t = 0.04 + (t - 6hours) * FT(-0.1 - 0.04) / FT(3hours)
    elseif t > FT(9hours) && t <= FT(12hours)
        ∂θ∂t_adv = FT(-0.08) + (t - 9hours) * FT(0.064) / FT(3hours)
        ∂θ∂t_rad = FT(0) 
        ∂qt∂t = -0.1 + (t - 9hours) * FT(-0.16 + 0.1) / FT(3hours)
    else
        ∂θ∂t_adv = FT(-0.016) 
        ∂θ∂t_rad = FT(0) + (t - 6hours) * FT(-0.1) / FT(2.5hours)
        ∂qt∂t = -0.16 + (t - 12hours) * FT(-0.3 + 0.16) / FT(2.5hours)
    end
        
    # Convert to [sec⁻¹] measures
    ∂θ∂t = (∂θ∂t_adv + ∂θ∂t_rad) ./ 1hours
    ∂qt∂t = ∂qt∂t ./ (1000 * 1hours)

    ρ∂θ∂t = state.ρ * ∂θ∂t
    ρ∂qt∂t = state.ρ * ∂qt∂t
    
    # Collect Sources
    source.moisture.ρq_tot += ρ∂qt∂t
    source.ρe += cvm * ρ∂θ∂t * exner(TS) + _e_int_v0 * ρ∂qt∂t
    source.ρe -= ρ * w_sub * dot(k̂, diffusive.∇h_tot)
    source.moisture.ρq_tot -= ρ * w_sub * dot(k̂, diffusive.moisture.∇q_tot)
    return nothing
end

function spline_init()
    ## Create Spline 
    z_in  = [0.0, 50.0,350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0 ] ## LES z is in meters
    θ_in  = [299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2] ## K
    rh_in = [15.2,15.17,14.98,14.8,14.7,13.5,3.0,3.0]/1000 ## relative humidity
    q_tot_in = rh_in ./ (1 .+ rh_in) ## total specific humidity 
 
    spl_θ = Spline1D(z_in, θ_in; k=1)
    spl_q_tot = Spline1D(z_in, q_tot_in; k=1)

    return (spl_θ = spl_θ, spl_q_tot = spl_q_tot)
end

"""
  Initial Condition for ARM LES
"""
function init_arm!(bl, state, aux, (x, y, z), t, args)
    # Problem floating point precision
    FT = eltype(state)
    
    # This experiment runs the ARM LES Configuration
    # (Continental Shallow-Cumulus cloud regime)
    # x,y,z imply eastward, northward and altitude coordinates in `[m]`

    # Unpack spline interpolant 
    # Extract spline objects from the optional configuration arguments
    
    spl_θ, spl_q_tot = args.spl_θ, args.spl_q_tot

    _grav = FT(grav(bl.param_set)) # Gravity - parameter set
    P_sfc::FT = 97000 # Surface air pressure
    rhg::FT = 15.2e-3 # Relative humidity at surface

    qg::FT = rhg ./ (1+rhg) # Total moisture at surface
    θ_liq_sfc = FT(299.0) # Prescribed θ_liq at surface

    q_pt_sfc = PhasePartition(qg) # Surface moisture partitioning

    Rm_sfc = gas_constant_air(bl.param_set, q_pt_sfc) # Moist gas constant

    T_sfc = FT(299.0) # Surface temperature 
    # TODO Change surface temperature via θ value ? 

    # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
    u::FT = 10
    v::FT = 0
    w::FT = 0

    # Get values for potential temperature 
    # and total specific humidity via interpolation
    θ_liq = FT(spl_θ(z))
    q_tot = FT(spl_q_tot(z))

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

    if z <= FT(200) # Add random perturbations to bottom 400m of model
        state.ρe += rand() * ρe_tot / 100
        state.moisture.ρq_tot += rand() * ρ * q_tot / 100
    end
end

#SHF = itp(-30, 90, 0, 4hours, t)
function energy_fluxes(state,aux,t)
    FT = eltype(state)
    hours = FT(3600)
    # Initialise
    SHF = FT(0)
    LHF = FT(0)
    # Time varying boundary condition
    if t <= FT(4hours)
        SHF = FT(-30) + FT(120)*t/(4hours)
        LHF = FT(5) + FT(245)*t/(4hours)
    elseif t > FT(4hours) && t <= FT(6.5hours)
        SHF = FT(90) + FT(50)*(t-4hours)/(2.5hours)
        LHF = FT(250) + FT(200)*(t-4hours)/(2.5hours)
    elseif t > FT(6.5hours) && t <= FT(7.5hours)
        SHF = FT(140) 
        LHF = FT(450) + FT(50)*(t-6.5hours)/(1hours)
    elseif t > FT(7.5hours) && t <= FT(10hours)
        SHF = FT(140) + FT(-40)*(t-7.5hours)/(2.5hours)
        LHF = FT(500) + FT(-80)*(t-7.5hours)/(2.5hours)
    elseif t > FT(10hours) && t <= FT(12.5hours)
        SHF = FT(100) + FT(-110)*(t-10hours)/(2.5hours)
        LHF = FT(420) + FT(-240)*(t-10hours)/(2.5hours)
    else
        SHF = FT(-10)
        LHF = FT(180) + FT(-180)*(t-12.5hours)/(2hours)
    end
    energy_flux = FT(LHF) .+ FT(SHF)
    return energy_flux
end


function moisture_fluxes(state,aux,t)
    FT = eltype(state)
    hours = FT(3600)
    LHF = FT(0)
    if t <= FT(4hours)
        LHF = FT(5) + FT(245)*(t-4hours)/(4hours)
    elseif t > FT(4hours) && t <= FT(6.5hours)
        LHF = FT(250) + FT(200)*(t-4hours)/(2.5hours)
    elseif t > FT(6.5hours) && t <= FT(7.5hours)
        LHF = FT(450) + FT(50)*(t-6.5hours)/(1hours)
    elseif t > FT(7.5hours) && t <= FT(10hours)
        LHF = FT(500) + FT(-80)*(t-7.5hours)/(2.5hours)
    elseif t > FT(10hours) && t <= FT(12.5hours)
        LHF = FT(420) + FT(-240)*(t-10hours)/(2.5hours)
    else
        LHF = FT(180) +  FT(-180)*(t-12.5hours)/(2hours)
    end
    T_sfc = FT(299)
    moisture_flux = LHF / latent_heat_vapor(param_set, T_sfc)
    return moisture_flux
end

function itp(y1,y2,x1,x2,x)
  result = (y1) + (y2-y1)*(x-x1)/(x2-x1)
end


function config_arm(FT, N, resolution, xmax, ymax, zmax)
    
    ics = init_arm!     # Initial conditions

    C_smag = FT(0.23)     # Smagorinsky coefficient

    u_star = FT(0.6)     # Friction velocity

    T_sfc = FT(299.0)     # Surface temperature `[K]`
    LHF = FT(147.2)       # Latent heat flux `[W/m²]`
    SHF = FT(9.5)         # Sensible heat flux `[W/m²]`
    moisture_flux = LHF / latent_heat_vapor(param_set, T_sfc)

    z_sponge = FT(3500)     # Start of sponge layer
    α_max = FT(0.75)        # Strength of sponge layer (timescale)
    γ = 2                   # Strength of sponge layer (exponent)

    u_geostrophic = FT(10)      # Eastward relaxation speed
    u_slope = FT(0)             # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)       # Northward relaxation speed

    f_coriolis = FT(8.5e-5) # Coriolis parameter
    

    # Assemble source components
    source = (
        Gravity(),
        ARMTendencies{FT}(),
        ARMSponge{FT}(
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        ARMGeostrophic{FT}(f_coriolis, u_geostrophic, u_slope, v_geostrophic),
    )

    # Choose default IMEX solver
    ode_solver_type = ClimateMachine.IMEXSolverType()
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
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> energy_fluxes(state,aux,t)),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) -> moisture_fluxes(state,aux,t),
                ),
            ),
            AtmosBC(),
        ),
        init_state_conservative = ics,
    )

    # Assemble configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "ARM-LES",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_arm!,
        solver_type = ode_solver_type,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    default_dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
    core_dgngrp = setup_atmos_core_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
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
    zmax = FT(4400)

    t0 = FT(0)

    # For a full-run, please set the timeend to 3600*6 seconds
    # For the test we set this to == 30 minutes
    #timeend = FT(1800)
    timeend = FT(52200)
    CFLmax = FT(0.90)

    splines = spline_init()
    
    driver_config = config_arm(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        splines;
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρe₀ = Q.ρe
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)
    cb_check_cons =
        GenericCallbacks.EveryXSimulationSteps(3000) do (init = false)
            Q = solver_config.Q
            δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
            δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
            @show (abs(δρ))
            @show (abs(δρe))
            nothing
        end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter, cb_check_cons),
        check_euclidean_distance = true,
    )
end

main()
