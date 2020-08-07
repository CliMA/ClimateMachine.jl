# A box advection test to visualise how different filters work

using MPI
using OrderedCollections
using Plots
using StaticArrays

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Mesh.Filters
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

import ClimateMachine.BalanceLaws:
    vars_state,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_prognostic!,
    boundary_state!

ClimateMachine.init(; disable_gpu = true, log_level = "warn");
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct Box1D{FT, _init_q, _amplitude, _velo} <: BalanceLaw
    param_set::AbstractParameterSet = param_set
    init_q::FT = _init_q
    amplitude::FT = _amplitude
    velo::FT = _velo
end

vars_state(::Box1D, ::Auxiliary, FT) = @vars(z_dim::FT);
vars_state(::Box1D, ::Prognostic, FT) = @vars(q::FT);
vars_state(::Box1D, ::Gradient, FT) = @vars();
vars_state(::Box1D, ::GradientFlux, FT) = @vars();

function init_state_auxiliary!(m::Box1D, aux::Vars, geom::LocalGeometry)
    aux.z_dim = geom.coord[3]
end;

function init_state_prognostic!(
    m::Box1D,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    if aux.z_dim >= 75 && aux.z_dim <= 125
        state.q = m.init_q + m.amplitude
    else
        state.q = m.init_q
    end
end;

function update_auxiliary_state!(
    dg::DGModel,
    m::Box1D,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return true
end;

function source!(m::Box1D, _...) end;

@inline function flux_first_order!(
    m::Box1D,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    FT = eltype(state)
    @inbounds begin
        flux.q = SVector(FT(0), FT(0), state.q * m.velo)
    end
end

@inline function flux_second_order!(
    m::Box1D,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) end

@inline function flux_second_order!(
    m::Box1D,
    flux::Grad,
    state::Vars,
    τ,
    d_h_tot,
) end

function boundary_state!(
    nf,
    m::Box1D,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) end;

FT = Float64;

function run_box1D(
    N_poly::Int,
    init_q::FT,
    amplitude::FT,
    velo::FT,
    plot_name::String;
    tmar_filter::Bool = false,
    cutoff_filter::Bool = false,
    exp_filter::Bool = false,
    boyd_filter::Bool = false,
    cutoff_param::Int = 1,
    exp_param_1::Int = 0,
    exp_param_2::Int = 32,
    boyd_param_1::Int = 0,
    boyd_param_2::Int = 32,
)
    N_poly = N_poly
    nelem = 128
    zmax = FT(350)

    m = Box1D{FT, init_q, amplitude, velo}()

    driver_config = ClimateMachine.SingleStackConfiguration(
        "Box1D",
        N_poly,
        nelem,
        zmax,
        param_set,
        m,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
        boundary = ((0, 0), (0, 0), (0, 0)),
        periodicity = (true, true, true),
    )

    t0 = FT(0)
    timeend = FT(450)

    Δ = min_node_distance(driver_config.grid, VerticalDirection())
    max_vel = m.velo
    dt = Δ / max_vel

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    grid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary

    output_dir = @__DIR__
    mkpath(output_dir)

    z_key = "z"
    z_label = "z"
    z = get_z(grid)

    all_data = Dict[dict_of_nodal_states(solver_config, [z_key])]  # store initial condition at ``t=0``
    time_data = FT[0]                                      # store time data

    # output
    step = [1]
    output_freq = floor(Int, timeend / dt) + 10

    cb_output = GenericCallbacks.EveryXSimulationSteps(output_freq) do
        push!(all_data, dict_of_nodal_states(solver_config, [z_key]))
        push!(time_data, gettime(solver_config.solver))
        nothing
    end

    filter_freq = 1
    # tmar filter
    cb_tmar =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:q,),
                solver_config.dg.grid,
                TMARFilter(),
            )
            nothing
        end
    # cutoff filter
    cb_cutoff =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:q,),
                solver_config.dg.grid,
                CutoffFilter(solver_config.dg.grid, cutoff_param),
            )
            nothing
        end
    # exponential filter
    cb_exp =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:q,),
                solver_config.dg.grid,
                ExponentialFilter(
                    solver_config.dg.grid,
                    exp_param_1,
                    exp_param_2,
                ),
            )
            nothing
        end
    # exponential filter
    cb_boyd =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:q,),
                solver_config.dg.grid,
                BoydVandevenFilter(
                    solver_config.dg.grid,
                    boyd_param_1,
                    boyd_param_2,
                ),
            )
            nothing
        end

    user_cb_arr = [cb_output]
    if tmar_filter
        push!(user_cb_arr, cb_tmar)
    end
    if cutoff_filter
        push!(user_cb_arr, cb_cutoff)
    end
    if exp_filter
        push!(user_cb_arr, cb_exp)
    end
    if boyd_filter
        push!(user_cb_arr, cb_boyd)
    end
    user_cb = (user_cb_arr...,)

    ClimateMachine.invoke!(solver_config; user_callbacks = (user_cb))

    push!(all_data, dict_of_nodal_states(solver_config, [z_key]))
    push!(time_data, gettime(solver_config.solver))

    export_plot(
        z,
        all_data,
        ("q",),
        joinpath(output_dir, plot_name);
        xlabel = "x",
        ylabel = "q",
        time_data = time_data,
        horiz_layout = true,
    )

end
