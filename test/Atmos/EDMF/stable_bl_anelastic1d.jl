using JLD2, FileIO
using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.BalanceLaws: vars_state
import ClimateMachine.BalanceLaws: projection
import ClimateMachine.DGMethods
using ClimateMachine.Atmos
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
import CLIMAParameters

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
# include("edmf_model.jl")
# include("edmf_kernels.jl")

# CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 290.0 # default
CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 265

function main(::Type{FT}) where {FT}
    # add a command line argument to specify the kind of surface flux
    # TODO: this will move to the future namelist functionality
    sbl_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(sbl_args, "StableBoundaryLayer")
    @add_arg_table! sbl_args begin
        "--surface-flux"
        help = "specify surface flux for energy and moisture"
        metavar = "prescribed|bulk|custom_sbl"
        arg_type = String
        default = "prescribed"
    end

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(1800)
    CFLmax = FT(100) # compared to soundwaves which are excluded in the Anelastic1D setup

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    N_updrafts = 1
    N_quad = 3
    turbconv = NoTurbConv()
    # compressibility = Compressible()
    compressibility = Anelastic1D()

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbulence = ConstantKinematicViscosity(FT(0)),
        turbconv = turbconv,
        compressibility = compressibility,
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_ANELASTIC_1D",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = FT(40),
        solver_type = ode_solver_type,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFLmax,
    )

    # --- Zero-out horizontal variations:
    vsp = vars_state(model, Prognostic(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :turbconv),
    )
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :energy, :ρe),
    )
    vsa = vars_state(model, Auxiliary(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        varsindex(vsa, :turbconv),
    )
    # ---

    dgn_config = config_diagnostics(driver_config)

    # boyd vandeven filter
    cb_boyd = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("energy.ρe",),
            solver_config.dg.grid,
            BoydVandevenFilter(
                solver_config.dg.grid,
                1, #default=0
                4, #default=32
            ),
        )
        nothing
    end

    diag_arr = [single_stack_diagnostics(solver_config)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 5
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            diag_vs_z = single_stack_diagnostics(solver_config)

            nstep = getsteps(solver_config.solver)

            push!(diag_arr, diag_vs_z)
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    # Mass tendencies = 0 for Anelastic1D model,
    # so mass should be completely conserved:
    check_cons =
        (ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.00000001)),)

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        user_callbacks = (cb_boyd, cb_data_vs_time, cb_print_step),
        check_euclidean_distance = true,
    )

    diag_vs_z = single_stack_diagnostics(solver_config)
    push!(diag_arr, diag_vs_z)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, diag_arr, time_data
end

solver_config, diag_arr, time_data = main(Float64)

# include(joinpath(@__DIR__, "report_mse_sbl_anelastic.jl"))

nothing
