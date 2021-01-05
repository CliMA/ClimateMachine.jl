# using FileIO
# using JLD2
using ClimateMachine
ClimateMachine.init(;
    parse_clargs = true,
    output_dir = get(ENV, "CLIMATEMACHINE_SETTINGS_OUTPUT_DIR", "output"),
    fix_rng_seed = true,
)
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.BalanceLaws: vars_state
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
using ClimateMachine.Atmos: AtmosModel
using ClimateMachine.Atmos: PressureGradientModel
using ClimateMachine.BalanceLaws
using ClimateMachine.Mesh.Filters: apply!
using ClimateMachine.DGMethods: AbstractCustomFilter, apply!
# import ClimateMachine.DGMethods: custom_filter!
import ClimateMachine.DGMethods: custom_filter!
using ClimateMachine.DGMethods: RemBL

include(joinpath(clima_dir, "experiments", "AtmosLES", "stable_bl_model.jl"))
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

using ClimateMachine.DGMethods: AbstractCustomFilter, apply!
struct EDMFFilter <: AbstractCustomFilter end
import ClimateMachine.DGMethods: custom_filter!

custom_filter!(f::EDMFFilter, bl::RemBL, state, aux) =
    custom_filter!(f, bl.main, state, aux)

function custom_filter!(::EDMFFilter, bl, state, aux)
    if hasproperty(bl, :turbconv)
        FT = eltype(state)
        # this ρu[3]=0 is only for single_stack
        state.ρu = SVector(state.ρu[1], state.ρu[2], 0)
        up = state.turbconv.updraft
        en = state.turbconv.environment
        N_up = n_updrafts(bl.turbconv)
        ρ_gm = state.ρ
        ρa_min = ρ_gm * bl.turbconv.subdomains.a_min
        ρa_max = ρ_gm - ρa_min
        ts = recover_thermo_state(bl, state, aux)
        θ_liq_gm = liquid_ice_pottemp(ts)
        ρaθ_liq_ups = sum(vuntuple(i -> up[i].ρaθ_liq, N_up))
        ρa_ups = sum(vuntuple(i -> up[i].ρa, N_up))
        ρaw_ups = sum(vuntuple(i -> up[i].ρaw, N_up))
        ρa_en = ρ_gm - ρa_ups
        ρaw_en = -ρaw_ups
        θ_liq_en = (θ_liq_gm - ρaθ_liq_ups) / ρa_en
        w_en = ρaw_en / ρa_en
        @unroll_map(N_up) do i
            # if !(ρa_min <= up[i].ρa <= ρa_max)
            #     up[i].ρa = min(max(up[i].ρa,ρa_min),ρa_max)
            #     up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
            #     up[i].ρaw     = FT(0)
            # end
            up[i].ρa = ρa_min
            up[i].ρaθ_liq = up[i].ρa * θ_liq_gm
            up[i].ρaw = FT(0)
        end
        en.ρatke = max(en.ρatke, FT(0))
        en.ρaθ_liq_cv = max(en.ρaθ_liq_cv, FT(0))
        # en.ρaq_tot_cv = max(en.ρaq_tot_cv,FT(0))
        # en.ρaθ_liq_q_tot_cv = max(en.ρaθ_liq_q_tot_cv,FT(0))
        # validate_variables(bl, state, aux, "custom_filter!")
    end

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

    cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sbl_args)

    surface_flux = cl_args["surface_flux"]

    # DG polynomial order
    N = 4
    nelem_vert = 20

    # Prescribe domain parameters
    zmax = FT(400)

    t0 = FT(0)

    # Simulation time
    timeend = FT(3600 * 6)
    CFLmax = FT(0.50)

    config_type = SingleStackConfigType

    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    N_updrafts = 1
    N_quad = 3 # Using N_quad = 1 leads to norm(Q) = NaN at init.
    turbconv = EDMF(FT, N_updrafts, N_quad)

    model = stable_bl_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbconv = turbconv,
    )

    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        "SBL_EDMF",
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
        varsindex(vsp, :ρe),
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
            (turbconv_filters(turbconv)...,),
            solver_config.dg.grid,
            TMARFilter(),
        )
        Filters.apply!( # comment this for NoTurbConv
            EDMFFilter(),
            solver_config.dg.grid,
            solver_config.dg.balance_law,
            solver_config.Q,
            solver_config.dg.state_auxiliary,
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
    state_types = (Prognostic(), Auxiliary(), GradientFlux())
    dons_arr = [dict_of_nodal_states(solver_config, state_types; interp = true)]
    time_data = FT[0]

    # Define the number of outputs from `t0` to `timeend`
    n_outputs = 5
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

    cb_check_cons = GenericCallbacks.EveryXSimulationSteps(3000) do
        Q = solver_config.Q
        δρ = (sum(Q.ρ .* M) - Σρ₀) / Σρ₀
        δρe = (sum(Q.ρe .* M) .- Σρe₀) ./ Σρe₀
        @show (abs(δρ))
        @show (abs(δρe))
        @test (abs(δρ) <= 0.001)
        @test (abs(δρe) <= 0.1)
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
    push!(dons_arr, dons)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, dons_arr, time_data, state_types
end

solver_config, dons_arr, time_data, state_types = main(Float64)

## Uncomment lines to save output using JLD2
#
# output_dir = @__DIR__;
# mkpath(output_dir);
# println(dons_arr[1].keys)
# z = get_z(solver_config.dg.grid; rm_dupes = true);
# save(
#     string(output_dir, "/sbl_edmf.jld2"),
#     "dons_arr",
#     dons_arr,
#     "time_data",
#     time_data,
#     "z",
#     z,
# )
