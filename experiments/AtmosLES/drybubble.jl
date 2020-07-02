#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using Distributions
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ----------------------------- Dry Bubble -------------------------- #
# 2-D rising bubble benchmark by Bryan and Fritsch 2002
# https://doi.org/10.1175/1520-0493(2002)130<2917:ABSFMN>2.0.CO;2
#
# 1) Domain: 20000km × 500m × 10000km invariant along y direction
# 2) Initialization:
#       Neutrally stratified environment θ=300K
#       Perturbed potential temperature centered at xc = 10000 zc=2000
#           with radius rc = 2000m
# 2) Boundary Conditions:
#
# Configuration defaults are in `src/Driver/Configurations.jl`

function init_drybubble!(bl, state, aux, (x, y, z), t)
    FT = eltype(state)

    ## parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    xc::FT = 10000
    zc::FT = 2000
    rc::FT = 2000
    L = sqrt(((x - xc)/rc)^2 + ((z - zc)/rc)^2)
    θamplitude::FT = 2
    θ_ref::FT = 300

    Δθ::FT = 0
    if L <= rc
        Δθ = θamplitude * (cos(L*π/2))^2
    end

    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)

    ρu = SVector(FT(0), FT(0), FT(0))                   # momentum
    ## State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl, aux)            # potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    ## Assign State Variables
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot

end

function config_drybubble(FT, N, resolution, xmax, ymax, zmax)
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    T_surface = FT(300)
    T_min_ref = FT(0)
    T_profile = DryAdiabaticProfile{FT}(param_set, T_surface, T_min_ref)
    ref_state = HydrostaticState(T_profile)

    _C_smag = FT(C_smag(param_set))
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = SmagorinskyLilly(_C_smag),
        moisture = DryModel(),
        hyperdiffusion = StandardHyperDiffusion(60),
        source = (Gravity(),),
        ref_state = ref_state,
        init_state_conservative = init_drybubble!,
    )


    config = ClimateMachine.AtmosLESConfiguration(
        "DryRisingBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_drybubble!,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    FT = Float64
    interval = "10000steps"
    boundaries = [
        FT(0.0) FT(0.0) FT(0.0)
        FT(20000) FT(500) FT(10000)
    ]
    resolution = (FT(100), FT(100), FT(100))
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )

    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    
    state_dgngrp = setup_dump_state_diagnostics(
        AtmosLESConfigType(),
        "1shours",
        driver_config.name,
        interpol = interpol,
    )
    aux_dgngrp = setup_dump_aux_diagnostics(
        AtmosLESConfigType(),
        "1shours",
        driver_config.name,
        interpol = interpol,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        dgngrp,state_dgngrp,aux_dgngrp,
    ])
end

function main()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(100)
    Δv = FT(100)
    resolution = (Δh, Δh, Δv)
    xmax = FT(20000)
    ymax = FT(500)
    zmax = FT(10000)
    t0 = FT(0)
    timeend = FT(1000)

    driver_config = config_drybubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
    )
    dgn_config = config_diagnostics(driver_config)

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (),
        check_euclidean_distance = true,
    )

    # Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
