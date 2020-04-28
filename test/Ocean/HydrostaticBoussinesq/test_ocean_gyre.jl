using Test
using CLIMA
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters
using CLIMA.VariableTemplates
using CLIMA.Mesh.Grids: polynomialorder
using CLIMA.HydrostaticBoussinesq

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

function test_ocean_gyre(; imex::Bool = false, BC = nothing, Δt = 60)
    CLIMA.init()

    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(4)
    Nʸ = Int(4)
    Nᶻ = Int(5)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 4e6    # m
    Lʸ = 4e6    # m
    H = 1000   # m
    dimensions = (Lˣ, Lʸ, H)

    timestart = FT(0)    # s
    timeend = FT(36000) # s

    if imex
        solver_type = CLIMA.IMEXSolverType(linear_model = LinearHBModel)
        Courant_number = 0.1
    else
        solver_type =
            CLIMA.ExplicitSolverType(solver_method = LSRK144NiegemannDiehlBusch)
        Courant_number = 0.4
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
        ode_solver_type = solver_type,
        ode_dt = FT(Δt),
        modeldata = modeldata,
        Courant_number = Courant_number,
    )

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
    ]

    for BC in boundary_conditions
        test_ocean_gyre(imex = false, BC = BC, Δt = 600)
        test_ocean_gyre(imex = true, BC = BC, Δt = 150)
    end
end
