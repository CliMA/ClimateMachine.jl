using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.DGMethods
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include("convective_bl_model.jl")

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    cbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(cbl_args, "ConvectiveBL")
    @add_arg_table! cbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "bulk"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = cbl_args)

    surface_flux = cl_args["surface_flux"]

    # FT = Float64
    config_type = SingleStackConfigType

    # DG polynomial order
    N = 3
    nelem_vert = 30

    # Prescribe domain parameters
    zmax = FT(3200)

    t0 = FT(0)

    timeend = FT(3600*6)
    CFLmax = FT(0.4)

    # Choose default IMEX solver
    ode_solver_type = ClimateMachine.IMEXSolverType()

    model = convective_bl_model(FT, config_type, zmax, surface_flux)
    ics = model.problem.init_state_prognostic
    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "CBL_SINGLE_STACK",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        solver_type = ode_solver_type,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    #################### Change the ode_solver to implicit solver
    # dg = solver_config.dg
    # Q = solver_config.Q
    # vdg = DGModel(
    #     driver_config;
    #     state_auxiliary = dg.state_auxiliary,
    #     direction = VerticalDirection(),
    # )
    # # linear solver relative tolerance rtol which should be slightly smaller than the nonlinear solver tol
    # linearsolver = BatchedGeneralizedMinimalResidual(
    #     dg,
    #     Q;
    #     max_subspace_size = 30,
    #     atol = -1.0,
    #     rtol = 5e-5,
    # )
    # """
    # N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat

    # F(Q) == 0
    # ||F(Q^i) || / ||F(Q^0) || < tol

    # """
    # # ϵ is a sensity parameter for this problem, it determines the finite difference Jacobian dF = (F(Q + ϵdQ) - F(Q))/ϵ
    # # I have also try larger tol, but tol = 1e-3 does not work
    # nonlinearsolver =
    #     JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4, ϵ = 1.e-10)

    # # this is a second order time integrator, to change it to a first order time integrator
    # # change it ARK1ForwardBackwardEuler, which can reduce the cost by half at the cost of accuracy 
    # # and stability
    # # preconditioner_update_freq = 50 means updating the preconditioner every 50 Newton solves, 
    # # update it more freqent will accelerate the convergence of linear solves, but updating it 
    # # is very expensive
    # ode_solver = ARK2ImplicitExplicitMidpoint( # try replace with Giraldo ...
    #     dg,
    #     vdg,
    #     NonLinearBackwardEulerSolver(
    #         nonlinearsolver;
    #         isadjustable = true,
    #         preconditioner_update_freq = 1, # set to 1 as most accurate and move on
    #     ),
    #     Q;
    #     dt = solver_config.dt,
    #     t0 = 0,
    #     split_explicit_implicit = false,
    #     variant = NaiveVariant(),
    # )

    # solver_config.solver = ode_solver

    #######################################
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρ₀ = Q.ρ
    ρe₀ = Q.ρe
    # DG variable sums
    Σρ₀ = sum(ρ₀ .* M)
    Σρe₀ = sum(ρe₀ .* M)

    grid = driver_config.grid

    # state_types = (Prognostic(), Auxiliary(), GradientFlux())
    state_types = (Prognostic(), Auxiliary())
    all_data = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 10
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(all_data,
                dict_of_nodal_states(solver_config, state_types; interp = true)
            )
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    cb_check_cons = GenericCallbacks.EveryXSimulationSteps(3000) do
        Q = solver_config.Q
        δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
        δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
        @show (abs(δρ))
        @show (abs(δρe))
        @test (abs(δρ) <= 0.001)
        @test (abs(δρe) <= 0.025)
        nothing
    end

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (
            cbtmarfilter,
            cb_check_cons,
            cb_data_vs_time,
            cb_print_step,
        ),
        check_euclidean_distance = true,
    )

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(all_data, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, all_data, time_data, state_types
end

solver_config, all_data, time_data, state_types = main(Float64)
