using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.SystemSolvers
using ClimateMachine.DGMethods
using ClimateMachine.SystemSolvers
import ClimateMachine.DGMethods: custom_filter!
using ClimateMachine.Mesh.Filters: apply!
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "experiments", "AtmosLES", "bomex_model.jl"))
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

    if m.moisture isa DryModel
        ρq_tot = FT(0)
        ts = PhaseDry(m.param_set, e_int, state.ρ)
    else
        ρq_tot = gm.moisture.ρq_tot
        ts = PhaseEquil(m.param_set, e_int, state.ρ, ρq_tot / state.ρ)
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

using ClimateMachine.DGMethods: AbstractCustomFilter, apply!
struct EDMFFilter <: AbstractCustomFilter end
import ClimateMachine.DGMethods: custom_filter!

function custom_filter!(::EDMFFilter, bl, state, aux)
    FT = eltype(state)
    a_min = bl.turbconv.subdomains.a_min
    up = state.turbconv.updraft
    N_up = n_updrafts(bl.turbconv)
    ρ_gm = state.ρ
    ts = recover_thermo_state(bl, state, aux)
    ρaθ_liq_ups = sum(vuntuple(i->up[i].ρaθ_liq, N_up))
    ρaq_tot_ups = sum(vuntuple(i->up[i].ρaq_tot, N_up))
    ρa_ups      = sum(vuntuple(i->up[i].ρa, N_up))
    ρaw_ups     = sum(vuntuple(i->up[i].ρaw, N_up))
    ρa_en        = ρ_gm - ρa_ups
    ρq_tot_gm   = state.moisture.ρq_tot
    ρaw_en      = - ρaw_ups
    ρaq_tot_en  = (ρq_tot_gm - ρaq_tot_ups) / ρa_en
    θ_liq_en    = (liquid_ice_pottemp(ts) - ρaθ_liq_ups) / ρa_en
    q_tot_en    = ρaq_tot_en / ρa_en
    w_en        = ρaw_en / ρa_en
    @unroll_map(N_up) do i
        a_up_mask = up[i].ρa < (ρ_gm * a_min)
        Δρ_area = max(a_up_mask * (ρ_gm * a_min - up[i].ρa), FT(0))
        up[i].ρa      += a_up_mask * Δρ_area
        up[i].ρaθ_liq += a_up_mask * θ_liq_en * Δρ_area
        up[i].ρaq_tot += a_up_mask * q_tot_en * Δρ_area
        up[i].ρaq_tot = max(up[i].ρaq_tot,FT(0))
        up[i].ρaw     += a_up_mask * w_en * Δρ_area
    end

struct ZeroVerticalVelocityFilter <: AbstractCustomFilter end
function custom_filter!(::ZeroVerticalVelocityFilter, bl, state, aux)
    state.ρu = SVector(state.ρu[1], state.ρu[2], 0)
end

function main(::Type{FT}) where {FT}
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

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = bomex_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(3000)

    t0 = FT(0)

    # Simulation time
    timeend = FT(400)
    timeend = FT(3600*5)
    # CFLmax = FT(0.90)
    CFLmax = FT(10.0)

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
    turbconv = EDMF(FT, N_updrafts, N_quad)

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
    dg = solver_config.dg
    Q = solver_config.Q
    vdg = DGModel(
        driver_config;
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
    )
    # linear solver relative tolerance rtol which should be slightly smaller than the nonlinear solver tol
    linearsolver = BatchedGeneralizedMinimalResidual(
        dg,
        Q;
        max_subspace_size = 30,
        atol = -1.0,
        rtol = 5e-5,
    )
    """
    N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat

    F(Q) == 0
    ||F(Q^i) || / ||F(Q^0) || < tol

    """
    # ϵ is a sensity parameter for this problem, it determines the finite difference Jacobian dF = (F(Q + ϵdQ) - F(Q))/ϵ
    # I have also try larger tol, but tol = 1e-3 does not work
    nonlinearsolver =
        JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-4, ϵ = 1.e-10)

    # this is a second order time integrator, to change it to a first order time integrator
    # change it ARK1ForwardBackwardEuler, which can reduce the cost by half at the cost of accuracy 
    # and stability
    # preconditioner_update_freq = 50 means updating the preconditioner every 50 Newton solves, 
    # update it more freqent will accelerate the convergence of linear solves, but updating it 
    # is very expensive
    ode_solver = ARK2ImplicitExplicitMidpoint(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = 50,
        ),
        Q;
        dt = solver_config.dt,
        t0 = 0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )

    solver_config.solver = ode_solver

    #######################################

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
        varsindex(vsp, :ρe),
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

    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot", turbconv_filters(turbconv)...),
            solver_config.dg.grid,
            TMARFilter(),
        )
        apply!(
            EDMFFilter(),
            solver_config.dg.grid,
            model,
            solver_config.Q,
            solver_config.dg.state_auxiliary,
        Filters.apply!(
            ZeroVerticalVelocityFilter(),
            solver_config.dg.grid,
            solver_config.dg.balance_law,
        )
        nothing
    end

    # state_types = (Prognostic(), Auxiliary(), GradientFlux())
    state_types = (Prognostic(), Auxiliary())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 10
    # This equates to exports every ceil(Int, timeend/n_outputs) time-step:
    every_x_simulation_time = ceil(Int, timeend / n_outputs)

    cb_data_vs_time =
        GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
            push!(
                dons_arr,
                dict_of_nodal_states(solver_config, state_types; interp = true),
            )
            push!(time_data, gettime(solver_config.solver))
            nothing
        end

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.001)),
        ClimateMachine.ConservationCheck("ρe", "3000steps", FT(0.0025)),
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

    dons = dict_of_nodal_states(solver_config, state_types; interp = true)
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)

include(joinpath(@__DIR__, "report_mse.jl"))
