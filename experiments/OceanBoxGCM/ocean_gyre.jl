#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init()
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.HydrostaticBoussinesq

using Test

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
    if BC == nothing
        problem = OceanGyre{FT}(dimensions...)
    else
        problem = OceanGyre{FT}(dimensions...; BC = BC)
    end

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = cʰ)

    config = ClimateMachine.OceanBoxGCMConfiguration(
        "ocean_gyre",
        N,
        resolution,
        model,
    )

    return config
end

function run_ocean_gyre(; imex::Bool = false, BC = nothing)
    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(20)
    Nʸ = Int(20)
    Nᶻ = Int(20)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    outpdir = "output"
    timestart = FT(0)    # s
    timeout = FT(0.25 * 86400) # s
    timeend = FT(86400) # s
    dt = FT(10)    # s

    if imex
        solver_type =
            ClimateMachine.IMEXSolverType(linear_model = LinearHBModel)
    else
        solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    end

    driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        # ode_dt = dt,
        Courant_number = 0.4,
        ode_solver_type = solver_type,
        modeldata = modeldata,
    )

    mkpath(outpdir)
    ClimateMachine.Settings.vtk = "never"
    # vtk_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.vtk = "$(vtk_interval)steps"

    ClimateMachine.Settings.diagnostics = "never"
    # diagnostics_interval = ceil(Int64, timeout / solver_config.dt)
    # ClimateMachine.Settings.diagnostics = "$(diagnostics_interval)steps"

    result = ClimateMachine.invoke!(solver_config)

    @test true
end

@testset "$(@__FILE__)" begin
    boundary_conditions = [
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineNoSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            ClimateMachine.HydrostaticBoussinesq.CoastlineFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            ClimateMachine.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
    ]

    for BC in boundary_conditions
        run_ocean_gyre(imex = false, BC = BC)
    end
end
