#!/usr/bin/env julia --project
#=
# This experiment file establishes the initial conditions, boundary conditions,
# source terms and simulation parameters (domain size + resolution) for
# a dry neutrally stratified Ekman layer.
# 
# The initial conditions are given by constant horizontal velocity of 1 m/s,
# and a constant potential temperature profile. The bottom boundary condition
# results in momentum drag, and there is no exchange of heat from the surface
# since the fluxes are zero and the temperature is homogeneous.
#
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
using CLIMAParameters.Planet: cp_d, cv_d, grav, T_surf_ref
using CLIMAParameters.Atmos.SubgridScale: C_smag, C_drag
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
import CLIMAParameters

using ClimateMachine.Atmos: altitude, recover_thermo_state, density

"""
  NeutralBL Geostrophic Forcing (Source)
"""
struct NeutralBLGeostrophic{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    "Coriolis parameter [s⁻¹]"
    f_coriolis::FT
    "Eastward geostrophic velocity `[m/s]` (Base)"
    u_geostrophic::FT
    "Eastward geostrophic velocity `[m/s]` (Slope)"
    u_slope::FT
    "Northward geostrophic velocity `[m/s]`"
    v_geostrophic::FT
end
NeutralBLGeostrophic(::Type{FT}, args...) where {FT} =
    NeutralBLGeostrophic{Momentum, FT}(args...)

function source(s::NeutralBLGeostrophic{Momentum}, m, args)
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
  NeutralBL Sponge (Source)
"""
struct NeutralBLSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
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

NeutralBLSponge(::Type{FT}, args...) where {FT} =
    NeutralBLSponge{Momentum, FT}(args...)

function source(s::NeutralBLSponge{Momentum}, m, args)
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
  Initial Condition for NeutralBoundaryLayer LES
"""
function init_problem!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    # Problem floating point precision
    param_set = parameter_set(bl)
    FT = eltype(state)
    c_p::FT = cp_d(param_set)
    c_v::FT = cv_d(param_set)
    _grav::FT = grav(param_set)
    γ::FT = c_p / c_v
    # Initialise speeds [u = Eastward, v = Northward, w = Vertical]
    u::FT = 1
    v::FT = 0
    w::FT = 0
    # Assign constant θ profile and equal to surface temperature
    θ::FT = T_surf_ref(param_set)

    p = aux.ref_state.p
    TS = PhaseDry_pθ(param_set, p, θ)

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
    add_perturbations!(state, localgeo)
    init_state_prognostic!(bl.turbconv, bl, state, aux, localgeo, t)
end

function ekman_layer_model(
    ::Type{FT},
    config_type,
    zmax,
    surface_flux;
    turbulence = ConstantKinematicViscosity(FT(0.1)),
    turbconv = NoTurbConv(),
    compressibility = Compressible(),
    ref_state = HydrostaticState(DryAdiabaticProfile{FT}(param_set),),
) where {FT}

    ics = init_problem!     # Initial conditions

    C_drag_::FT = C_drag(param_set) # FT(0.001)    # Momentum exchange coefficient
    u_star = FT(0.30)
    z_0 = FT(0.1)          # Roughness height

    z_sponge = FT(300)     # Start of sponge layer
    α_max = FT(0.75)       # Strength of sponge layer (timescale)
    γ = 2                  # Strength of sponge layer (exponent)

    u_geostrophic = FT(1)        # Eastward relaxation speed
    u_slope = FT(0)              # Slope of altitude-dependent relaxation speed
    v_geostrophic = FT(0)        # Northward relaxation speed
    f_coriolis = FT(1.39e-4) # Coriolis parameter at 73N

    q_sfc = FT(0)
    θ_sfc = T_surf_ref(param_set)
    g = compressibility isa Compressible ? (Gravity(),) : ()

    # Assemble source components
    source_default = (
        g...,
        NeutralBLSponge(
            FT,
            zmax,
            z_sponge,
            α_max,
            γ,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        NeutralBLGeostrophic(
            FT,
            f_coriolis,
            u_geostrophic,
            u_slope,
            v_geostrophic,
        ),
        turbconv_sources(turbconv)...,
    )
    source = source_default

    # Set up problem initial and boundary conditions
    if surface_flux == "prescribed"
        energy_bc = PrescribedEnergyFlux((state, aux, t) -> FT(0))
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> FT(0))
    elseif surface_flux == "bulk"
        energy_bc = BulkFormulaEnergy(
            (bl, state, aux, t, normPu_int) -> C_drag_,
            (bl, state, aux, t) -> (θ_sfc, q_sfc),
        )
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag_,
            (state, aux, t) -> q_sfc,
        )
    elseif surface_flux == "custom_sbl"
        energy_bc = PrescribedTemperature((state, aux, t) -> θ_sfc)
        moisture_bc = BulkFormulaMoisture(
            (state, aux, t, normPu_int) -> C_drag_,
            (state, aux, t) -> q_sfc,
        )
    elseif surface_flux == "Nishizawa2018"
        energy_bc = NishizawaEnergyFlux(
            (bl, state, aux, t, normPu_int) -> z_0,
            (bl, state, aux, t) -> (θ_sfc, q_sfc),
        )
        moisture_bc = PrescribedMoistureFlux((state, aux, t) -> FT(0))
    else
        @warn @sprintf(
            """
%s: unrecognized surface flux; using 'prescribed'""",
            surface_flux,
        )
    end

    moisture_bcs = ()
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
        moisture = DryModel(),
        source = source,
        turbconv = turbconv,
        compressibility = compressibility,
    )

    return model
end

function config_diagnostics(driver_config)
    default_dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        "60ssecs",
        driver_config.name,
    )
    core_dgngrp = setup_atmos_core_diagnostics(
        AtmosLESConfigType(),
        "60ssecs",
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        default_dgngrp,
        core_dgngrp,
    ])
end