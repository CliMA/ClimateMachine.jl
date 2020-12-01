using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.Atmos: AtmosModel
using ClimateMachine.Atmos: vars_state
using ClimateMachine.Atmos: PressureGradientModel
using ClimateMachine.BalanceLaws: vars_state
using ClimateMachine.Checkpoint
using ClimateMachine.DGMethods
import ClimateMachine.DGMethods: custom_filter!
using ClimateMachine.DGMethods: RemBL
import ClimateMachine.DGMethods: custom_filter!, rhs_prehook_filters
using ClimateMachine.Mesh.Filters: apply!
using ClimateMachine.ODESolvers
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using ClimateMachine.BalanceLaws
using ClimateMachine.SingleStackUtils
using ClimateMachine.SystemSolvers
# import ClimateMachine.DGMethods: custom_filter!

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"))
ENV["CLIMATEMACHINE_SETTINGS_DISABLE_GPU"] = true
ENV["CLIMATEMACHINE_SETTINGS_MONITOR_COURANT_NUMBERS"] = "3000steps"
ENV["CLIMATEMACHINE_SETTINGS_MONITOR_TIMESTEP_DURATION"] = "3000steps"
ENV["CLIMATEMACHINE_SETTINGS_FIX_RNG_SEED"] = true
include("stable_bl_model.jl")

using ClimateMachine.DGMethods: AbstractCustomFilter, apply!
rhs_prehook_filters(atmos::BalanceLaw) = MyCustomFilter()

struct MyCustomFilter <: AbstractCustomFilter end
function custom_filter!(::MyCustomFilter, balance_law, state, aux)
    # state.ρu = SVector(state.ρu[1],state.ρu[2],0)
end

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    sbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(sbl_args, "StableBoundaryLayer")
    @add_arg_table! sbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk"
        arg_type = String
        default = "bulk"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    config_type = SingleStackConfigType

    # DG polynomial order
    N = 4

    # Prescribe domain parameters
    nelem_vert = 20
    zmax = FT(400)

    t0 = FT(0)

    timeend = FT(3600 * 6)
    # timeend = FT(3600 * 6) # goal for 10 min run

    use_explicit_stepper_with_small_Δt = false
    if use_explicit_stepper_with_small_Δt
        CFLmax = FT(0.9)
        # ode_solver_type = ClimateMachine.IMEXSolverType()

        ode_solver_type = ClimateMachine.ExplicitSolverType(
            solver_method = LSRK144NiegemannDiehlBusch,
        )
    else
        CFLmax = FT(25)
        ode_solver_type = ClimateMachine.IMEXSolverType(
            implicit_model = AtmosAcousticGravityLinearModel,
            implicit_solver = SingleColumnLU,
            solver_method = ARK2GiraldoKellyConstantinescu,
            split_explicit_implicit = true,
            # split_explicit_implicit = false,
            discrete_splitting = false,
            # discrete_splitting = true,
        )
    # isothermal zonal flow
    # ode_solver_type = ClimateMachine.IMEXSolverType(
    #     implicit_model = AtmosAcousticGravityLinearModel,
    #     implicit_solver = ManyColumnLU,
    #     solver_method = ARK2GiraldoKellyConstantinescu,
    #     split_explicit_implicit = false,
    #     discrete_splitting = true,
    # )
    end

    model = stable_bl_model(FT, config_type, zmax, surface_flux)
    ics = model.problem.init_state_prognostic
    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_SINGLE_STACK",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        solver_type = ode_solver_type,
        hmax = zmax,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )
    dgn_config = config_diagnostics(driver_config)

    state_types = (Prognostic(), Auxiliary())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 100
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(dons_arr, dict_of_nodal_states(solver_config, state_types; interp = true))
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            (),
            solver_config.dg.grid,
            TMARFilter(),
        )
        Filters.apply!(
            MyCustomFilter(),
            solver_config.dg.grid,
            solver_config.dg.balance_law,
            solver_config.Q,
            solver_config.dg.state_auxiliary,
        )
        nothing
    end

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "3000steps", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cbtmarfilter, cb_data_vs_time),
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        check_euclidean_distance = true,
    )

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))
    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)

export_state_plots(
    solver_config,
    dons_arr,
    time_data,
    joinpath("output", "sbl_ss_acc");
    z = Array(get_z(solver_config.dg.grid; rm_dupes = true)),
)

export_state_contours(
    solver_config,
    dons_arr,
    time_data,
    joinpath("output", "sbl_ss_acc");
    z = Array(get_z(solver_config.dg.grid; rm_dupes = true)),
)