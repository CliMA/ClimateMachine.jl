#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.MPIStateArrays: euclidean_distance, norm
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: polynomialorder
using ClimateMachine.Ocean.HydrostaticBoussinesq

using Test

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

function config_simple_box(FT, N, resolution, dimensions; BC = nothing)
    if BC == nothing
        problem = SimpleBox{FT}(dimensions...)
    else
        problem = SimpleBox{FT}(dimensions...; BC = BC)
    end

    model = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem;
        cʰ = FT(1),
        αᵀ = FT(0),
        κʰ = FT(0),
        κᶻ = FT(0),
        fₒ = FT(0),
        β = FT(0),
    )

    config = ClimateMachine.OceanBoxGCMConfiguration(
        "hydrostatic_spindown",
        N,
        resolution,
        param_set,
        model;
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )

    return config
end

function run_hydrostatic_test(; imex::Bool = false, BC = nothing, refDat = ())
    FT = Float64

    # DG polynomial order
    N = Int(4)

    # Domain resolution and size
    Nˣ = Int(5)
    Nʸ = Int(5)
    Nᶻ = Int(8)
    resolution = (Nˣ, Nʸ, Nᶻ)

    Lˣ = 1e6    # m
    Lʸ = 1e6    # m
    H = 400   # m
    dimensions = (Lˣ, Lʸ, H)

    timestart = FT(0)   # s
    timeout = FT(6 * 3600)  # s
    timeend = FT(86400) # s
    dt = FT(120)    # s

    if imex
        solver_type =
            ClimateMachine.IMEXSolverType(implicit_model = LinearHBModel)
    else
        solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    end

    driver = config_simple_box(FT, N, resolution, dimensions; BC = BC)

    grid = driver.grid
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)
    modeldata = (vert_filter = vert_filter, exp_filter = exp_filter)

    solver = ClimateMachine.SolverConfiguration(
        timestart,
        timeend,
        driver,
        init_on_cpu = true,
        ode_dt = dt,
        ode_solver_type = solver_type,
        modeldata = modeldata,
    )

    output_interval = ceil(Int64, timeout / solver.dt)

    ClimateMachine.Settings.vtk = "never"
    # ClimateMachine.Settings.vtk = "$(output_interval)steps"

    ClimateMachine.Settings.diagnostics = "never"
    # ClimateMachine.Settings.diagnostics = "$(output_interval)steps"

    cb = ClimateMachine.StateCheck.sccreate(
        [(solver.Q, "state"), (solver.dg.state_auxiliary, "aux")],
        output_interval;
        prec = 12,
    )

    result = ClimateMachine.invoke!(solver; user_callbacks = [cb])

    Q_exact = ClimateMachine.DGMethods.init_ode_state(
        solver.dg,
        timeend,
        solver.init_args...;
        init_on_cpu = solver.init_on_cpu,
    )

    error = euclidean_distance(solver.Q, Q_exact) / norm(Q_exact)

    println("error = ", error)
    @test isapprox(error, FT(0.0); atol = 0.005)

    ## Check results against reference
    ClimateMachine.StateCheck.scprintref(cb)
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cb, refDat)
    end
end

@testset "$(@__FILE__)" begin

    include("../refvals/3D_hydrostatic_spindown_refvals.jl")

    run_hydrostatic_test(imex = false, refDat = refVals.explicit) # error = 0.0011289879366523504
    run_hydrostatic_test(imex = true, refDat = refVals.imex)  # error = 0.0033063071773607243
end
