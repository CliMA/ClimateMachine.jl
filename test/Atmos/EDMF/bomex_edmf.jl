using ClimateMachine
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.DGMethods
using ClimateMachine.SystemSolvers
import ClimateMachine.DGMethods: custom_filter!
using ClimateMachine.Mesh.Filters: apply!
using ClimateMachine.BalanceLaws: vars_state
using JLD2, FileIO
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "experiments", "AtmosLES", "bomex_model.jl"))
include(joinpath("helper_funcs", "diagnostics_configuration.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

"""
    init_state_prognostic!(
            turbconv::EDMF{FT},
            m::AtmosModel{FT},
            state::Vars,
            aux::Vars,
            localgeo,
            t::Real,
        ) where {FT}

Initialize EDMF state variables.
This method is only called at `t=0`.
"""
function init_state_prognostic!(
    turbconv::EDMF{FT},
    m::AtmosModel{FT},
    state::Vars,
    aux::Vars,
    localgeo,
    t::Real,
) where {FT}
    # Aliases:
    gm = state
    en = state.turbconv.environment
    up = state.turbconv.updraft
    N_up = n_updrafts(turbconv)
    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w)
    z = altitude(m, aux)

    # SCM setting - need to have separate cases coded and called from a folder - see what LES does
    # a moist_thermo state is used here to convert the input θ,q_tot to e_int, q_tot profile
    e_int = internal_energy(m, state, aux)
    param_set = parameter_set(m)
    if moisture_model(m) isa DryModel
        ρq_tot = FT(0)
        ts = PhaseDry(param_set, e_int, state.ρ)
    else
        ρq_tot = gm.moisture.ρq_tot
        ts = PhaseEquil_ρeq(param_set, state.ρ, e_int, ρq_tot / state.ρ)
    end
    T = air_temperature(ts)
    p = air_pressure(ts)
    q = PhasePartition(ts)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = ρq_tot * a_min
    end

    # initialize environment covariance with zero for now
    if z <= FT(2500)
        en.ρatke = gm.ρ * (FT(1) - z / FT(3000))
    else
        en.ρatke = FT(0)
    end
    en.ρaθ_liq_cv = FT(1e-5) / max(z, FT(10))
    en.ρaq_tot_cv = FT(1e-5) / max(z, FT(10))
    en.ρaθ_liq_q_tot_cv = FT(1e-7) / max(z, FT(10))
    return nothing
end;

struct ZeroVerticalVelocityFilter <: AbstractCustomFilter end
function custom_filter!(::ZeroVerticalVelocityFilter, bl, state, aux)
    state.ρu = SVector(state.ρu[1], state.ρu[2], 0)
end

function main(::Type{FT}, cl_args) where {FT}

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(3000)

    t0 = FT(0)

    # Simulation time
    timeend = FT(400)
    CFLmax = FT(1.2)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.IMEXSolverType(
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = SingleColumnLU,
        solver_method = ARK2GiraldoKellyConstantinescu,
        split_explicit_implicit = true,
        discrete_splitting = false,
    )

    N_updrafts = 1
    N_quad = 3
    turbconv = EDMF(FT, N_updrafts, N_quad, param_set)

    model =
        bomex_model(FT, config_type, zmax, surface_flux; turbconv = turbconv)

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "BOMEX_EDMF",
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = zmax,
    )

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_solver_type = ode_solver_type,
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
    horizontally_average!(
        driver_config.grid,
        solver_config.Q,
        varsindex(vsp, :moisture, :ρq_tot),
    )

    vsa = vars_state(model, Auxiliary(), FT)
    horizontally_average!(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        varsindex(vsa, :turbconv),
    )
    # ---

    dgn_config =
        config_diagnostics(driver_config, timeend; interval = "50ssecs")

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot", turbconv_filters(turbconv)...),
            solver_config.dg.grid,
            TMARFilter(),
        )
        Filters.apply!(
            ZeroVerticalVelocityFilter(),
            solver_config.dg.grid,
            solver_config.dg.balance_law,
            solver_config.Q,
            solver_config.dg.state_auxiliary,
        )
        nothing
    end

    diag_arr = [single_stack_diagnostics(solver_config)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 10
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            diag_vs_z = single_stack_diagnostics(solver_config)

            nstep = getsteps(solver_config.solver)
            # Save to disc (for debugging):
            # @save "bomex_edmf_nstep=$nstep.jld2" diag_vs_z

            push!(diag_arr, diag_vs_z)
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.001)),
        ClimateMachine.ConservationCheck("energy.ρe", "3000steps", FT(0.0025)),
    )

    cb_print_step = GenericCallbacks.EveryXSimulationSteps(100) do
        @show getsteps(solver_config.solver)
        nothing
    end

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        user_callbacks = (cbtmarfilter, cb_data_vs_time, cb_print_step),
        check_euclidean_distance = true,
    )

    diag_vs_z = single_stack_diagnostics(solver_config)
    push!(diag_arr, diag_vs_z)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, diag_arr, time_data
end


# add a command line argument to specify the kind of surface flux
# TODO: this will move to the future namelist functionality
bomex_args = ArgParseSettings(autofix_names = true)
add_arg_group!(bomex_args, "BOMEX")
@add_arg_table! bomex_args begin
    "--surface-flux"
    help = "specify surface flux for energy and moisture"
    metavar = "prescribed|bulk"
    arg_type = String
    default = "prescribed"
end

cl_args = ClimateMachine.init(
    parse_clargs = true,
    custom_clargs = bomex_args,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)

solver_config, diag_arr, time_data = main(Float64, cl_args)

include(joinpath(@__DIR__, "report_mse_bomex.jl"))

nothing
