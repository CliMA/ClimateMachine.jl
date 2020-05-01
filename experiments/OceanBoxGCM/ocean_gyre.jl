#!/usr/bin/env julia --project
using CLIMA
CLIMA.init()
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.VariableTemplates
using CLIMA.Mesh.Grids: polynomialorder
using CLIMA.HydrostaticBoussinesq

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

    config = CLIMA.OceanBoxGCMConfiguration("ocean_gyre", N, resolution, model)

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
        solver_type = CLIMA.IMEXSolverType(linear_model = LinearHBModel)
    else
        solver_type =
            CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
    end

    driver_config = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver_config = CLIMA.SolverConfiguration(
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
    CLIMA.Settings.vtk = "never"
    # vtk_interval = ceil(Int64, timeout / solver_config.dt)
    # CLIMA.Settings.vtk = "$(vtk_interval)steps"

    CLIMA.Settings.diagnostics = "never"
    # diagnostics_interval = ceil(Int64, timeout / solver_config.dt)
    # CLIMA.Settings.diagnostics = "$(diagnostics_interval)steps"

    result = CLIMA.invoke!(solver_config)

    @test true
end

@testset "$(@__FILE__)" begin
    boundary_conditions = [
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineNoSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
        (
            CLIMA.HydrostaticBoussinesq.CoastlineFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanFloorFreeSlip(),
            CLIMA.HydrostaticBoussinesq.OceanSurfaceNoStressForcing(),
        ),
    ]

    for BC in boundary_conditions
        run_ocean_gyre(imex = false, BC = BC)
    end
end
