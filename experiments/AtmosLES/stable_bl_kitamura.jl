
#!/usr/bin/env julia --project

using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
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

import ClimateMachine.BalanceLaws: vars_state_conservative, vars_state_auxiliary
import ClimateMachine.Atmos: source!, atmos_source!, altitude
import ClimateMachine.Atmos: flux_second_order!, thermo_state

"""
  StableBL Geostrophic Forcing (Source)
"""
struct StableBLGeostrophic{FT} <: Source
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
    s::StableBLGeostrophic,
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
  StableBL Sponge (Source)
"""
struct StableBLSponge{FT} <: Source
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
    s::StableBLSponge,
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
  StableBLTendencies (Source)
Moisture, Temperature and Subsidence tendencies
"""
struct StableBLTendencies{FT} <: Source
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
    s::StableBLTendencies,
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
function init_nishizawa_sf!(bl, state, aux, (x, y, z), t)
    # This experiment runs the BOMEX LES Configuration
    # (Shallow cumulus cloud regime)
    # x,y,z imply eastward, northward and altitude coordinates in `[m]`

    # Problem floating point precision
    FT = eltype(state)

    P_sfc::FT = 1.015e5 # Surface air pressure
    qg::FT = 22.45e-3 # Total moisture at surface
    q_pt_sfc = PhasePartition(qg) # Surface moisture partitioning
    Rm_sfc = gas_constant_air(bl.param_set, q_pt_sfc) # Moist gas constant
    θ_liq_sfc = FT(265) # Prescribed θ_liq at surface
    T_sfc = FT(300.4) # Surface temperature
    _grav = FT(grav(bl.param_set))

    # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
    u::FT = 8
    v::FT = 0
    w::FT = 0

    # Assign piecewise quantities to θ_liq and q_tot
    θ_liq::FT = 0
    q_tot::FT = 0

    # Piecewise functions for potential temperature and total moisture
    z1 = FT(100)
    if z <= z1
        θ_liq = FT(265)
    else
        θ_liq = FT(265) + FT(0.01)*(z − z1)
    end

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

    if z <= FT(50) # Add random perturbations to bottom 50m of model
        state.ρe += rand() * ρe_tot / 100
        state.moisture.ρq_tot += rand() * ρ * q_tot / 100
    end
end

function config_nishizawa_sf(FT, N, resolution, xmax, ymax, zmax)

    ics = init_nishizawa_sf!     # Initial conditions

    C_smag = FT(0.23)     # Smagorinsky coefficient

    u_star = FT(0.28)     # Friction velocity

    T_sfc = FT(300.4)     # Surface temperature `[K]`
    LHF = FT(147.2)       # Latent heat flux `[W/m²]`
    SHF = FT(9.5)         # Sensible heat flux `[W/m²]`
    moisture_flux = LHF / latent_heat_vapor(param_set, T_sfc)

    ∂qt∂t_peak = FT(-1.2e-8)  # Moisture tendency (energy source)
    zl_moisture = FT(300)     # Low altitude limit for piecewise function (moisture source)
    zh_moisture = FT(500)     # High altitude limit for piecewise function (moisture source)
    ∂θ∂t_peak = FT(-2 / FT(day(param_set)))  # Potential temperature tendency (energy source)

    z_sponge = FT(300)     # Start of sponge layer
    α_max = FT(0.75)       # Strength of sponge layer (timescale)
    γ = 2                  # Strength of sponge layer (exponent)

    u_geostrophic = FT(8)        # Eastward relaxation speed
    u_slope = FT(0)              # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)        # Northward relaxation speed

    f_coriolis = FT(1.39e-4) # Coriolis parameter

    # Assemble source components
    source = (
        Gravity(),
        StableBLSponge{FT}(
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        StableBLGeostrophic{FT}(f_coriolis, u_geostrophic, u_slope, v_geostrophic),
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
        init_nishizawa_sf!,
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
    Δh = FT(20)
    Δv = FT(20)

    resolution = (Δh, Δh, Δv)

    # Prescribe domain parameters
    xmax = FT(400)
    ymax = FT(400)
    zmax = FT(400)

    t0 = FT(0)

    # For a full-run, please set the timeend to 3600*6 seconds
    # For the test we set this to == 30 minutes
    timeend = FT(3600 * 9)
    CFLmax = FT(0.90)

    driver_config = config_nishizawa_sf(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
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
    cb_check_cons = GenericCallbacks.EveryXSimulationSteps(3000) do
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
