using JLD2

# using Logging: disable_logging, Warn
# disable_logging(Warn)

function setup_solver(nlalg::IterativeAlgorithm, driver_config, t0, timeend, dt)
    # for implicit ARK timestepper; specify nonlinear solver
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    dg = solver_config.dg
    Q = solver_config.Q
    vdg = DGModel(
        driver_config;
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )

    ode_solver = ARK2GiraldoKellyConstantinescu(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nlalg;
            isadjustable = true,
            preconditioner_update_freq = -1,
        ),
        Q;
        dt = dt,
        t0 = t0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )
    solver_config.solver = ode_solver
    return solver_config
end

# function setup_solver(implicitsolver, driver_config, t0, timeend, dt)
#     # for implicit ARK timestepper; specify nonlinear solver
#     solver_config = ClimateMachine.SolverConfiguration(
#         t0,
#         timeend,
#         driver_config,
#         ode_dt = dt,
#     )

#     dg = solver_config.dg
#     Q = solver_config.Q
#     vdg = DGModel(
#         driver_config;
#         state_auxiliary = dg.state_auxiliary,
#         direction = VerticalDirection(),
#     )

#     ode_solver = implicitsolver(
#         dg,
#         vdg,
#         NonLinearBackwardEulerSolver(
#             JacobianFreeNewtonKrylovAlgorithm(
#             GeneralizedMinimalResidualAlgorithm(;
#                 preconditioner = ColumnwiseLUPreconditioningAlgorithm(update_period = 50),
#                 M = 50, atol = 1e-6, rtol = 1e-6,
#             );
#             maxiters = Int(2e3), atol = 1e-6, rtol = 1e-6, autodiff=false);
#             isadjustable = true,
#             preconditioner_update_freq = 100,
#         ),
#         Q;
#         dt = dt,
#         t0 = t0,
#         split_explicit_implicit = false,
#         variant = NaiveVariant(),
#     )
#     solver_config.solver = ode_solver
#     return solver_config
# end

function setup_solver(explicitsolver, driver_config, t0, timeend, dt)
    # for explicit timesteppers
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    dg = solver_config.dg
    Q = solver_config.Q

    ode_solver = explicitsolver(
        dg,
        Q;
        dt = dt,
        t0 = t0,
    )
    solver_config.solver = ode_solver
    return solver_config
end

# load "true" data and create interpolation function
@load joinpath(@__DIR__, "infiltration_truth.jld2") truedata # a dict_of_nodal_states
true_moisture_continuous = Spline1D(truedata["z"][:], truedata["soil.water.ϑ_l"][:])

"""
    benchmark_solver_at_dt!(solverkey::String, datadict::Dict, driver_config, t0, timeend, dt)

Benchmark model specified by `driver_config` with timestep `dt` and solver algorithm specified by `datdict[solverkey]`.
"""
function benchmark_solver_at_dt!(solverkey::String, datadict::Dict, driver_config, t0, timeend, dt)
    solveralg = datadict[solverkey].algorithm
    solver_config = setup_solver(solveralg, driver_config, t0, timeend, dt)

    # solver run
    solvetime = @elapsed ClimateMachine.invoke!(solver_config)
    @info solvetime
    GC.gc()
    if solvetime < 6
        samples = 10
    elseif solvetime < 12
        samples = 5
    else
        samples = 1
    end

    for i = 1:(samples-1)
        solver_config = setup_solver(solveralg, driver_config, t0, timeend, dt)
        solvetime += @elapsed ClimateMachine.invoke!(solver_config)
        @info solvetime
        GC.gc()
    end
    solvetime /= samples

    finalstate = dict_of_nodal_states(solver_config; interp = true)  # store final state at `timeend`

    # interpolation & error
    current_profile = finalstate["soil.water.ϑ_l"][:]
    simulation_z = finalstate["z"][:]
    true_profile = true_moisture_continuous.(simulation_z)
    rmse = sqrt(sum((true_profile .- current_profile).^2.0))
    
    append!(datadict[solverkey].dts, dt)
    append!(datadict[solverkey].solvetimes, solvetime)
    append!(datadict[solverkey].rmse, rmse)
    append!(datadict[solverkey].finalstates, finalstate)
end

"""
    benchmark_solver_at_dts!(solverkey::String, datadict::Dict, driver_config, t0, timeend, dts)

Appends benchmark run at each timestep in `dts` to existing dictionary entry for `solverkey`.

If a `dt` in `dts` previously has a stored entry, the trial is skipped.
"""
function benchmark_solver_at_dts!(solverkey::String, datadict::Dict, driver_config, t0, timeend, dts)
    for dt in dts
        if !(dt in datadict[solverkey].dts) # don't rerun if previously solved at that timestep
            println("Solver: $solverkey, dt: $dt")
            benchmark_solver_at_dt!(solverkey, datadict, driver_config, t0, timeend, dt)
        end
    end
end

"""
    benchmark_solvers(solvers, driver_config, t0, timeend, dts)

Run infiltration test on each solver in `solvers` at each dt in `dts`.

Appends new runs to existing dictionary of benchmarking data.
"""
function benchmark_solvers!(solvers, driver_config, datadict, t0, timeend, dts)
    for (i, solveralg) in enumerate(solvers)
        solverkey = string(keys(solvers)[i])
        if !(solverkey in keys(datadict))
            push!(datadict, solverkey => (algorithm = solveralg, dts = [], solvetimes = [], rmse = [], finalstates = []))
        end
        benchmark_solver_at_dts!(solverkey, datadict, driver_config, t0, timeend, dts)
    end
end

"""
    benchmark_solvers(solvers, driver_config, t0, timeend, dts)

Run infiltration test on each solver in `solvers` at each dt in `dts`.

Returns a dictionary of named tuples containing relevant information.
"""
function benchmark_solvers(solvers, driver_config, t0, timeend, dts)
    datadict = Dict{String, NamedTuple}()
    benchmark_solvers!(solvers, driver_config, datadict, t0, timeend, dts)
    return datadict
end

using Cassette
using Cassette: @context, @overdub, prehook

mutable struct Count
    dgcalls::Int
    gmresiters::Int
    picarditers::Int
    newtoniters::Int
end

@context CountCtx
Cassette.prehook(ctx::CountCtx{Count}, f::DGModel, args...) = ctx.metadata.dgcalls += 1
Cassette.prehook(ctx::CountCtx{Count}, f, arg::StandardPicardSolver, bool::Bool) = ctx.metadata.picarditers += 1
Cassette.prehook(ctx::CountCtx{Count}, f, arg::JaCobIanfrEEneWtONKryLovSoLVeR, threshold, iters, oters) = ctx.metadata.newtoniters += 1
Cassette.prehook(ctx::CountCtx{Count}, f, arg::GeneralizedMinimalResidualSolver, threshold, iters, oters) = ctx.metadata.gmresiters += 1
Cassette.prehook(ctx::CountCtx{Count}, f, arg::BatchedGeneralizedMinimalResidualSolver, threshold, iters, oters) = ctx.metadata.gmresiters += 1

function callcount(solver, driver_config, t0, timeend, dts)
    for dt in dts
        solver_config = setup_solver(solver, driver_config, t0, timeend, dt)
        c = Count(0,0,0,0)
        @overdub(CountCtx(metadata = c), ClimateMachine.invoke!(solver_config))
        @info "dt: $dt, Counts = $c"
        GC.gc()
    end
end