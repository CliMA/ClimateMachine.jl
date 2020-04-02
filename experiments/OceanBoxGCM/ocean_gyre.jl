using Test
using CLIMA
using CLIMA.HydrostaticBoussinesq
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.Mesh.Filters

using CLIMA.VariableTemplates
using CLIMA.Mesh.Grids: polynomialorder

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(FT, N, resolution, dimensions)
    prob = OceanGyre{FT}(dimensions...)

    _grav::FT = grav(param_set)

    cʰ = sqrt(_grav * prob.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(prob, cʰ = cʰ)

    config = CLIMA.OceanBoxGCMConfiguration("ocean_gyre", N, resolution, model)

    return config
end

function run_ocean_gyre(; imex::Bool = false)
    CLIMA.init()

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

    driver_config = config_simple_box(FT, N, resolution, dimensions)

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
        Courant_number = 0.25,
        ode_solver_type = solver_type,
        modeldata = modeldata,
    )

    mkpath(outpdir)
    CLIMA.Settings.enable_vtk = false
    CLIMA.Settings.vtk_interval = ceil(Int64, timeout / solver_config.dt)

    CLIMA.Settings.enable_diagnostics = false
    CLIMA.Settings.diagnostics_interval =
        ceil(Int64, timeout / solver_config.dt)

    result = CLIMA.invoke!(solver_config)

end

run_ocean_gyre(imex = false)
