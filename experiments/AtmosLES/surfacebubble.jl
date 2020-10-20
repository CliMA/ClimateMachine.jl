#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using Distributions
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# -------------------- Surface Driven Bubble ----------------- #
# Rising thermals driven by a prescribed surface heat flux.
# 1) Boundary Conditions:
#       Laterally periodic with no flow penetration through top
#       and bottom wall boundaries.
#       Momentum: Impenetrable(FreeSlip())
#       Energy:   Spatially varying non-zero heat flux up to time t₁
# 2) Domain: 1250m × 1250m × 1000m
# Configuration defaults are in `src/Driver/Configurations.jl`


"""
  Surface Driven Thermal Bubble
"""
function init_surfacebubble!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord
    FT = eltype(state)
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    xc::FT = 1250
    yc::FT = 1250
    zc::FT = 1250
    θ_ref::FT = 300
    Δθ::FT = 0

    #Perturbed state:
    θ = θ_ref + Δθ # potential temperature
    π_exner = FT(1) - _grav / (c_p * θ) * z # exner pressure
    ρ = p0 / (R_gas * θ) * (π_exner)^(c_v / R_gas) # density

    q_tot = FT(0)
    ts = PhaseEquil_ρθq(bl.param_set, ρ, θ, q_tot)
    q_pt = PhasePartition(ts)

    ρu = SVector(FT(0), FT(0), FT(0))
    # energy definitions
    e_kin = FT(0)
    e_pot = gravitational_potential(bl.orientation, aux)
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)
    state.ρ = ρ
    state.ρu = ρu
    state.ρe = ρe_tot
    state.moisture.ρq_tot = ρ * q_pt.tot
end

function config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
    # Boundary conditions
    # Heat Flux Peak Magnitude
    F₀ = FT(100)
    # Time [s] at which `heater` turns off
    t₁ = FT(500)
    # Plume wavelength scaling
    x₀ = xmax
    function energyflux(state, aux, t)
        x = aux.coord[1]
        y = aux.coord[2]
        MSEF = F₀ * (cospi(2 * x / x₀))^2 * (cospi(2 * y / x₀))^2
        t < t₁ ? MSEF : zero(MSEF)
    end

    C_smag = FT(0.23)

    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    problem = AtmosProblem(
        boundarycondition = (
            AtmosBC(energy = PrescribedEnergyFlux(energyflux)),
            AtmosBC(),
        ),
        init_state_prognostic = init_surfacebubble!,
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        turbulence = SmagorinskyLilly{FT}(C_smag),
        moisture = EquilMoist{FT}(),
        source = (Gravity(),),
    )
    config = ClimateMachine.AtmosLESConfiguration(
        "SurfaceDrivenBubble",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_surfacebubble!,
        solver_type = ode_solver,
        model = model,
    )

    return config
end

function config_diagnostics(
    driver_config,
    xmax::FT,
    ymax::FT,
    zmax::FT,
    resolution,
) where {FT}
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    boundaries = [
        FT(0) FT(0) FT(0)
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    pdgngrp = setup_atmos_default_perturbations(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
        interpol = interpol,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp, pdgngrp])
end

function main()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(50)
    Δv = FT(50)
    resolution = (Δh, Δh, Δv)
    xmax = FT(2000)
    ymax = FT(2000)
    zmax = FT(2000)
    t0 = FT(0)
    timeend = FT(2000)

    driver_config = config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
    )
    dgn_config = config_diagnostics(driver_config, xmax, ymax, zmax, resolution)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

    @test isapprox(result, FT(1); atol = 1.5e-3)
end

main()
