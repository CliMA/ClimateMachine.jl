using Test
using ClimateMachine
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.BalanceLaws
using ClimateMachine.Ocean.HydrostaticBoussinesq
using ClimateMachine.Ocean.OceanProblems

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(name, resolution, dimensions, problem; BC = nothing)
    if BC == nothing
        problem = problem{FT}(dimensions...)
    else
        problem = problem{FT}(dimensions...; BC = BC)
    end

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(param_set, problem, cʰ = cʰ)

    N, Nˣ, Nʸ, Nᶻ = resolution
    resolution = (Nˣ, Nʸ, Nᶻ)

    config = ClimateMachine.OceanBoxGCMConfiguration(
        name,
        N,
        resolution,
        param_set,
        model,
    )

    return config
end

function run_simple_box(
    name,
    resolution,
    dimensions,
    timespan,
    problem;
    imex::Bool = false,
    BC = nothing,
    Δt = nothing,
    refDat = (),
)
    if imex
        solver_type =
            ClimateMachine.IMEXSolverType(implicit_model = LinearHBModel)
        Courant_number = 0.1
    else
        solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
        Courant_number = 0.4
    end

    driver_config =
        config_simple_box(name, resolution, dimensions, problem; BC = BC)

    grid = driver_config.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    timestart, timeend = timespan
    solver_config = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver_config,
        init_on_cpu = true,
        ode_solver_type = solver_type,
        ode_dt = Δt,
        modeldata = modeldata,
        Courant_number = Courant_number,
    )

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    nt_freq = floor(Int, 1 // 10 * solver_config.timeend / solver_config.dt)
    cb = ClimateMachine.StateCheck.sccreate(
        [(solver_config.Q, "Q"), (solver_config.dg.state_auxiliary, "s_aux")],
        nt_freq;
        prec = 12,
    )

    result = ClimateMachine.invoke!(solver_config; user_callbacks = [cb])

    ## Check results against reference if present
    ClimateMachine.StateCheck.scprintref(cb)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cb, refDat)
    end
end
