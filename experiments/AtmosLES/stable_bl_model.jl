#!/usr/bin/env julia --project
#=
# This experiment file establishes the initial conditions, boundary conditions,
# source terms and simulation parameters (domain size + resolution) for the
# GABLS LES case ([Beare2006](@cite); [Kosovic2000](@cite)).
#
## [Kosovic2000](@cite)
#
# To simulate the experiment, type in
#
# julia --project experiments/AtmosLES/stable_bl_les.jl
=#

using ArgParse
using Distributions
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Printf
using UnPack

using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.ODESolvers
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.TurbulenceConvection
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws: source

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav, day
using CLIMAParameters.Atmos.SubgridScale: C_smag, C_drag
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
import CLIMAParameters

using ClimateMachine.Atmos: altitude, recover_thermo_state, density, pressure

"""
  StableBL Geostrophic Forcing (Source)
"""
struct StableBLGeostrophic{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    "Coriolis parameter [s⁻¹]"
    f_coriolis::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end
StableBLGeostrophic(::Type{FT}, args...) where {FT} =
    StableBLGeostrophic{Momentum, FT}(args...)

function source(s::StableBLGeostrophic{Momentum}, m, args)
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
  StableBL Sponge (Source)
"""
struct StableBLSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
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

StableBLSponge(::Type{FT}, args...) where {FT} =
    StableBLSponge{Momentum, FT}(args...)

function source(s::StableBLSponge{Momentum}, m, args)
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

add_perturbations!(state, localgeo) = nothing

"""
  Initial Condition for StableBoundaryLayer LES
"""
function init_problem!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    # Problem floating point precision
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v
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
        θ_liq = FT(265) + FT(0.01) * (z - z1)
    end

    θ = θ_liq
    p = aux.ref_state.p
    # Establish thermodynamic state and moist phase partitioning
    if bl.moisture isa DryModel
        TS = PhaseDry_pθ(bl.param_set, p, θ)
    else
        TS = PhaseEquil_pθq(bl.param_set, p, θ_liq, q_tot)
    end

    ρ = bl.compressibility isa Compressible ? air_density(TS) : aux.ref_state.ρ

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
    state.energy.ρe = ρe_tot
    if !(bl.moisture isa DryModel)
        state.moisture.ρq_tot = ρ * q_tot
    end
    add_perturbations!(state, localgeo)
    init_state_prognostic!(bl.turbconv, bl, state, aux, localgeo, t)
end

function surface_temperature_variation(state, t)
    FT = eltype(state)
    return FT(265) - FT(1 / 4) * (t / 3600)
end

function stable_bl_model(
    ::Type{FT},
    config_type,
    zmax,
    surface_flux;
    turbulence = ConstantKinematicViscosity(FT(0)),
    turbconv = NoTurbConv(),
    compressibility = Compressible(),
    moisture_model = "dry",
    ref_state = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
) where {FT}

    ics = init_problem!     # Initial conditions

    C_drag_::FT = C_drag(param_set) # FT(0.001)    # Momentum exchange coefficient
    u_star = FT(0.30)
    z_0 = FT(0.1)          # Roughness height

    z_sponge = FT(300)     # Start of sponge layer
    α_max = FT(0.75)       # Strength of sponge layer (timescale)
    γ = 2                  # Strength of sponge layer (exponent)

    u_geostrophic = FT(8)        # Eastward relaxation speed
    u_slope = FT(0)              # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)        # Northward relaxation speed
    f_coriolis = FT(1.39e-4) # Coriolis parameter at 73N

    q_sfc = FT(0)

    g = compressibility isa Compressible ? (Gravity(),) : ()
    LHF = FT(0) # Latent heat flux `[W/m²]`
    SHF = FT(0) # Sensible heat flux `[W/m²]`

    # Assemble source components
    source_default = (
        g...,
        StableBLSponge(
            FT,
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        # StableBLGeostrophic(
        #     FT,
        #     f_coriolis,
        #     u_geostrophic,
        #     u_slope,
        #     v_geostrophic,
        # ),
        turbconv_sources(turbconv)...,
    )
    if moisture_model == "dry"
        source = source_default
        moisture = DryModel()
    elseif moisture_model == "equilibrium"
        source = source_default
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(0.1))
    elseif moisture_model == "nonequilibrium"
        source = (source_default..., CreateClouds()...)
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        source = source_default
    end
    # Set up problem initial and boundary conditions
    if surface_flux == "prescribed"
        energy_bc = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF)
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> moisture_flux)
        # energy_bc = PrescribedEnergyFlux((state, aux, t) -> FT(0))
        # moisture_bc = PrescribedMoistureFlux((state, aux, t) -> FT(0))
    elseif surface_flux == "bulk"
        energy_bc = BulkFormulaEnergy(
            (bl, state, aux, t, normPu_int) -> C_drag_,
            (bl, state, aux, t) ->
                (surface_temperature_variation(state, t), q_sfc),
        )
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag_,
            (state, aux, t) -> q_sfc,
        )
    elseif surface_flux == "custom_sbl"
        energy_bc = PrescribedTemperature(
            (state, aux, t) -> surface_temperature_variation(state, t),
        )
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag_,
            (state, aux, t) -> q_sfc,
        )
    elseif surface_flux == "Nishizawa2018"
        energy_bc = NishizawaEnergyFlux(
            (bl, state, aux, t, normPu_int) -> z_0,
            (bl, state, aux, t) ->
                (surface_temperature_variation(state, t), q_sfc),
        )
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> moisture_flux)
    else
        @warn @sprintf(
            """
%s: unrecognized surface flux; using 'prescribed'""",
            surface_flux,
        )
    end

    moisture_bcs = moisture_model == "dry" ? () : (; moisture = moisture_bc)
    boundary_conditions = (
        AtmosBC(;
            momentum = Impenetrable(DragLaw(
                # normPu_int is the internal horizontal speed
                # P represents the projection onto the horizontal
                (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
            )),
            energy = energy_bc,
            moisture_bcs...,
            turbconv = turbconv_bcs(turbconv)[1],
        ),
        AtmosBC(; turbconv = turbconv_bcs(turbconv)[2]),
    )

    moisture_flux = FT(0)
    problem = AtmosProblem(
        init_state_prognostic = ics,
        boundaryconditions = boundary_conditions,
    )

    # Assemble model components
    model = AtmosModel{FT}(
        config_type,
        param_set;
        problem = problem,
        ref_state = ref_state,
        turbulence = turbulence,
        moisture = moisture,
        source = source,
        turbconv = turbconv,
        compressibility = compressibility,
    )

    return model
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
