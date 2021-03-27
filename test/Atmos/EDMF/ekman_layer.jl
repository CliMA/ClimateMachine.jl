#!/usr/bin/env julia --project
#=
# This driver file simulates the ekman_layer_model.jl in a single column setting.
# 
# The user may select in main() the following configurations:
# - DG or FV vertical discretization by changing the boolean `finite_volume`,
# - Compressible() or Anelastic1D() changing the compressibility,
# - Constant kinematic viscosity, Smagorinsky-Lilly or EDMF SGS fluxes.
#
# The default is DG, Anelastic1D(), constant kinematic viscosity of 0.1. 
#
=#

using JLD2, FileIO
using ClimateMachine
using ClimateMachine.SingleStackUtils
using ClimateMachine.Checkpoint
using ClimateMachine.BalanceLaws: vars_state
using ClimateMachine.Atmos
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
import CLIMAParameters
import ClimateMachine.DGMethods.FVReconstructions: FVLinear

using ClimateMachine.Mesh.Filters: AbstractFilterTarget
using ClimateMachine.Orientations: gravitational_potential
using ClimateMachine.Thermodynamics: PhaseDry_ρp, PhaseEquil_ρpq, total_energy
import ClimateMachine.Mesh.Filters: vars_state_filtered,
    compute_filter_argument!, compute_filter_result!
using ClimateMachine.Mesh.Filters: apply!, BoydVandevenFilter


include(joinpath(clima_dir, "experiments", "AtmosLES", "ekman_layer_model.jl"))
include(joinpath("helper_funcs", "diagnostics_configuration.jl"))
include("edmf_model.jl")
include("edmf_kernels.jl")

CLIMAParameters.Planet.T_surf_ref(::EarthParameterSet) = 290.0
CLIMAParameters.Atmos.EDMF.a_surf(::EarthParameterSet) = 0.0
function set_clima_parameters(filename)
    eval(:(include($filename)))
end

struct EnergyPerturbation{M} <: AbstractFilterTarget
    atmos::M
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

Initialize EDMF state variables if turbconv=EDMF(...) is selected.
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
    z = altitude(m, aux)

    param_set = parameter_set(m)
    ts = new_thermo_state(m, state, aux)
    θ_liq = liquid_ice_pottemp(ts)

    a_min = FT(0) #turbconv.subdomains.a_min
    @unroll_map(N_up) do i
        up[i].ρa = gm.ρ * a_min
        up[i].ρaw = gm.ρu[3] * a_min
        up[i].ρaθ_liq = gm.ρ * a_min * θ_liq
        up[i].ρaq_tot = FT(0)
    end

    en.ρatke =
        z > FT(250) ? FT(0) :
        gm.ρ *
        FT(0.4) *
        FT(1 - z / 250.0) *
        FT(1 - z / 250.0) *
        FT(1 - z / 250.0)
    en.ρaθ_liq_cv = FT(0)
    en.ρaq_tot_cv = FT(0)
    en.ρaθ_liq_q_tot_cv = FT(0)
    return nothing
end;
#dennis 
vars_state_filtered(target::EnergyPerturbation, FT) = @vars(e_tot::FT)

ref_thermo_state(atmos::AtmosModel, aux::Vars) =
    ref_thermo_state(atmos, aux, atmos.moisture)
ref_thermo_state(atmos::AtmosModel, aux::Vars, ::DryModel) =
    PhaseDry_ρp(
        parameter_set(atmos),
        aux.ref_state.ρ,
        aux.ref_state.p,
    )
ref_thermo_state(atmos::AtmosModel, aux::Vars, ::Any) =
    PhaseEquil_ρpq(
        parameter_set(atmos),
        aux.ref_state.ρ,
        aux.ref_state.p,
        aux.ref_state.ρq_tot / aux.ref_state.ρ,
    )
function compute_filter_argument!(
    target::EnergyPerturbation,
    filter_state::Vars,
    state::Vars,
    aux::Vars,
)
    filter_state.e_tot = state.energy.ρe / state.ρ
    filter_state.e_tot -= total_energy(
        zero(eltype(aux)),
        gravitational_potential(target.atmos, aux),
        ref_thermo_state(target.atmos, aux),
    )
end
function compute_filter_result!(
    target::EnergyPerturbation,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
)
    filter_state.e_tot += total_energy(
        zero(eltype(aux)),
        gravitational_potential(target.atmos, aux),
        ref_thermo_state(target.atmos, aux),
    )
    state.energy.ρe = state.ρ * filter_state.e_tot
end
#dennis 
function main(::Type{FT}, cl_args) where {FT}
    # Change boolean to control vertical discretization type
    finite_volume = false

    # Choice of compressibility and CFL
    # compressibility = Compressible()
    compressibility = Anelastic1D()
    str_comp = compressibility == Compressible() ? "COMPRESS" : "ANELASTIC"

    # Choice of SGS model
    # turbconv = NoTurbConv()
    N_updrafts = 1
    N_quad = 3
    turbconv = EDMF(FT, N_updrafts, N_quad, param_set)

    C_smag_ = C_smag(param_set)
    turbulence = ConstantKinematicViscosity(FT(0))
    # turbulence = SmagorinskyLilly{FT}(C_smag_)

    # Prescribe domain parameters
    zmax = FT(400)
    # Simulation time
    t0 = FT(0)
    timeend = FT(1800) # Change to 7h for low-level jet
    CFLmax = compressibility == Compressible() ? FT(1) : FT(100)

    config_type = SingleStackConfigType
    ode_solver_type = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    if finite_volume
        N = (1, 0)
        nelem_vert = 80
        ref_state = HydrostaticState(
            DryAdiabaticProfile{FT}(param_set),
            ;
            subtract_off = false,
        )
        output_prefix = string("EL_", str_comp, "_FVM")
        fv_reconstruction = FVLinear()
    else
        N = 4
        nelem_vert = 20
        ref_state = HydrostaticState(DryAdiabaticProfile{FT}(param_set),)
        output_prefix = string("EL_", str_comp, "_DG")
        fv_reconstruction = nothing
    end

    surface_flux = cl_args["surface_flux"]
    model = ekman_layer_model(
        FT,
        config_type,
        zmax,
        surface_flux;
        turbulence = turbulence,
        turbconv = turbconv,
        ref_state = ref_state,
        compressibility = compressibility,
    )
    # Assemble configuration
    driver_config = ClimateMachine.SingleStackConfiguration(
        output_prefix,
        N,
        nelem_vert,
        zmax,
        param_set,
        model;
        hmax = FT(40),
        solver_type = ode_solver_type,
        fv_reconstruction = fv_reconstruction,
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

    # dennis
    #####################################################################
    cb_filter = GenericCallbacks.EveryXSimulationSteps(1) do
        apply!(
            solver_config.Q,
            EnergyPerturbation(driver_config.bl),
            driver_config.grid,
            BoydVandevenFilter(driver_config.grid, 1, 4);
            state_auxiliary = solver_config.dg.state_auxiliary,
            direction = VerticalDirection(),
        )
    end
    # dennis





    # boyd vandeven filter
    cb_boyd = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            # ("energy.ρe",),
            (turbconv_filters(turbconv)...,),
            solver_config.dg.grid,
            BoydVandevenFilter(solver_config.dg.grid, 1, 4),
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
    Δρ_lim = compressibility == Compressible() ? FT(0.001) : FT(0.00000001)
    check_cons = (ClimateMachine.ConservationCheck("ρ", "3000steps", Δρ_lim),)

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
        user_callbacks = (cb_filter, cb_data_vs_time, cb_print_step),
        check_euclidean_distance = true,
    )

    diag_vs_z = single_stack_diagnostics(solver_config)
    push!(diag_arr, diag_vs_z)
    push!(time_data, gettime(solver_config.solver))

    return solver_config, diag_arr, time_data
end

# ArgParse in global scope to modify Clima Parameters
sl_args = ArgParseSettings(autofix_names = true)
add_arg_group!(sl_args, "EkmanLayer")
@add_arg_table! sl_args begin
    "--cparam-file"
    help = "specify CLIMAParameters file"
    arg_type = Union{String, Nothing}
    default = nothing

    "--surface-flux"
    help = "specify surface flux for energy and moisture"
    metavar = "prescribed|bulk|custom_sl"
    arg_type = String
    default = "prescribed"
end

cl_args = ClimateMachine.init(parse_clargs = true, custom_clargs = sl_args)
if !isnothing(cl_args["cparam_file"])
    filename = cl_args["cparam_file"]
    set_clima_parameters(filename)
end

solver_config, diag_arr, time_data = main(Float64, cl_args)

nothing
