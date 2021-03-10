using JLD2, FileIO
using ClimateMachine
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.BalanceLaws: vars_state
import ClimateMachine.BalanceLaws: projection
import ClimateMachine.DGMethods
using ClimateMachine.Atmos
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
import CLIMAParameters

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

# CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 290.0 # default
CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 265

function set_clima_parameters(filename)
    eval(:(include($filename)))
end

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
    # a thermo state is used here to convert the input θ to e_int profile
    e_int = internal_energy(m, state, aux)

    ts = PhaseDry(m.param_set, e_int, state.ρ)
    T = air_temperature(ts)
    p = air_pressure(ts)
    q = PhasePartition(ts)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = FT(0)
    end

    # initialize environment covariance with zero for now
    if z <= FT(250)
        en.ρatke =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
        en.ρaθ_liq_cv =
            gm.ρ *
            FT(0.4) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0) *
            FT(1 - z / 250.0)
    else
        en.ρatke = FT(0)
        en.ρaθ_liq_cv = FT(0)
    end
    en.ρaq_tot_cv = FT(0)
    en.ρaθ_liq_q_tot_cv = FT(0)
    return nothing
end;

function main(::Type{FT}, cl_args) where {FT}

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(1800 * 1)
    CFLmax = FT(100)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    N_updrafts = 1
    N_quad = 3
    turbconv = NoTurbConv()
    # turbconv = EDMF(FT, N_updrafts, N_quad, param_set)
    # compressibility = Compressible()
    compressibility = Anelastic1D()
    C_smag_ = C_smag(param_set)

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbulence = SmagorinskyLilly{FT}(C_smag_),
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

    if !isnothing(cl_args["cparam_file"])
        ClimateMachine.Settings.output_dir = cl_args["cparam_file"] * ".output"
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

# ArgParse in global scope to modify Clima Parameters
sbl_args = ArgParseSettings(autofix_names = true)
add_arg_group!(sbl_args, "StableBoundaryLayer")
@add_arg_table! sbl_args begin
    "--cparam-file"
    help = "specify CLIMAParameters file"
    arg_type = Union{String, Nothing}
    default = nothing

    "--surface-flux"
    help = "specify surface flux for energy and moisture"
    metavar = "prescribed|bulk|custom_sbl"
    arg_type = String
    default = "custom_sbl"
end

cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)
if !isnothing(cl_args["cparam_file"])
    filename = cl_args["cparam_file"]
    set_clima_parameters(filename)
end

solver_config, diag_arr, time_data = main(Float64, cl_args)

include(joinpath(@__DIR__, "report_mse_sbl_anelastic.jl"))

nothing
