using MPI
using OrderedCollections
using StaticArrays
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
import ClimateMachine.DGMethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

FT = Float64;

ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

include(joinpath(clima_dir, "docs", "plothelpers.jl"));

Base.@kwdef struct HeatModel{FT} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Heat capacity"
    ρc::FT = 1
    "Thermal diffusivity"
    α::FT = 0.01
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

m = HeatModel{FT}();

vars_state_auxiliary(::HeatModel, FT) = @vars(z::FT, T::FT);
vars_state_conservative(::HeatModel, FT) = @vars(ρcT::FT);
vars_state_gradient(::HeatModel, FT) = @vars(ρcT::FT);
vars_state_gradient_flux(::HeatModel, FT) = @vars(α∇ρcT::SVector{3, FT});

function init_state_auxiliary!(m::HeatModel, aux::Vars, geom::LocalGeometry)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;
function init_state_conservative!(m::HeatModel,state::Vars,aux::Vars,coords,t::Real)
    state.ρcT = m.ρc * aux.T
end;

function update_auxiliary_state!(dg::DGModel,m::HeatModel,Q,t::Real,elems::UnitRange)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
end;
function heat_eq_nodal_update_aux!(m::HeatModel,state::Vars,aux::Vars,t::Real)
    aux.T = state.ρcT / m.ρc
end;
function compute_gradient_argument!(m::HeatModel,transform::Vars,state::Vars,aux::Vars,t::Real)
    transform.ρcT = state.ρcT
end;
function compute_gradient_flux!(m::HeatModel,diffusive::Vars,∇transform::Grad,state::Vars,aux::Vars,t::Real)
    diffusive.α∇ρcT = -m.α * ∇transform.ρcT
end;
function source!(m::HeatModel, _...) end;
function flux_first_order!(m::HeatModel,flux::Grad,state::Vars,aux::Vars,t::Real,) end;
function flux_second_order!(m::HeatModel,flux::Grad,state::Vars,diffusive::Vars,hyperdiffusive::Vars,aux::Vars,t::Real)
    flux.ρcT += diffusive.α∇ρcT
end;

function boundary_state!(nf,m::HeatModel,state⁺::Vars,aux⁺::Vars,n⁻,state⁻::Vars,aux⁻::Vars,bctype,t,_...)
    bctype == 1 && (state⁺.ρcT = m.ρc * m.T_bottom)
end;

function boundary_state!(nf,m::HeatModel,state⁺::Vars,diff⁺::Vars,aux⁺::Vars,n⁻,state⁻::Vars,diff⁻::Vars,aux⁻::Vars,bctype,t,_...)
    bctype == 1 && (state⁺.ρcT = m.ρc * m.T_bottom)
    bctype == 2 && (diff⁺.α∇ρcT = n⁻ * m.flux_top)
end;

n_poly = 5;
nelem_vert = 10;
zmax = FT(1);

driver_config = ClimateMachine.SingleStackConfiguration("HeatEquation", n_poly, nelem_vert, zmax, param_set, m, numerical_flux_first_order = CentralNumericalFluxFirstOrder());

t0 = FT(0)
timeend = FT(40)
Δ = min_node_distance(driver_config.grid)
given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / m.α;
dt = Fourier_bound

sc = ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

using DiffEqBase

# function DiffEqJLSolverConfiguration(sc, alg, args...; kwargs...)
#     increment = false
#     # (du, u, p, t) -> rhs_implicit!(du, u, p, t; increment = false),
#     # (du, u, p, t) -> rhs!(du, u, p, t; increment = false),
#     rhs_implicit!(du, u, p, t) = sc.dg(sc.solver.dQ, sc.Q, nothing, t, increment)
#     # rhs!(du, u, p, t) = sc.dg(sc.solver.dQ, sc.Q, nothing, t, increment)
#     rhs!(du, u, p, t) = nothing # purely implicit RHS
#     solver = DiffEqJLIMEXSolver(rhs!, rhs_implicit!, alg, sc.Q, args..., sc.t0; kwargs...)
#     return ClimateMachine.SolverConfiguration(sc.name,
#     sc.mpicomm,
#     sc.param_set,
#     sc.dg,
#     sc.Q,
#     sc.t0,
#     sc.timeend,
#     sc.dt,
#     sc.init_on_cpu,
#     sc.numberofsteps,
#     sc.init_args,
#     solver)
# end

using OrdinaryDiffEq: Rosenbrock23
# sc = DiffEqJLSolverConfiguration(sc, Rosenbrock23(); dt=sc.dt)

sc = ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);
using OrdinaryDiffEq
prob = ODEProblem(sc.dg, sc.Q, (0.0, sc.timeend),nothing)
# solve(prob,Rosenbrock23(autodiff=false),
#             save_everystep = false,
#             save_start = false,
#             save_end = false,)
solve(prob,Kvaerno3(),
            save_everystep = false,
            save_start = false,
            save_end = false,)

# grid = sc.dg.grid;
# Q = sc.Q;
# aux = sc.dg.state_auxiliary;

# output_dir = @__DIR__;

# mkpath(output_dir);

# z_scale = 100 # convert from meters to cm
# z_key = "z"
# z_label = "z [cm]"
# z = get_z(grid, z_scale)
# st_cons = vars_state_conservative(m, FT)
# st_aux = vars_state_auxiliary(m, FT)
# state_vars = get_vars_from_nodal_stack(grid,Q,st_cons)
# aux_vars = get_vars_from_nodal_stack(grid,aux,st_aux)
# all_vars = OrderedDict(state_vars..., aux_vars...);
# f = joinpath(output_dir, "initial_condition.png")
# export_plot_snapshot(z, all_vars, ("ρcT",), f, z_labe);

# const n_outputs = 5;
# const every_x_simulation_time = ceil(Int, timeend / n_outputs);
# all_data = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
# all_data[1] = all_vars # store initial condition at ``t=0``

# callback = GenericCallbacks.EveryXSimulationTime(
#     every_x_simulation_time,
#     sc.solver,
# ) do (init = false)
#     state_vars = get_vars_from_nodal_stack(grid,Q,st_cons)
#     aux_vars = get_vars_from_nodal_stack(grid,aux,st_aux;exclude = [z_key])
#     push!(all_data, OrderedDict(state_vars..., aux_vars...))
#     nothing
# end;

# ClimateMachine.invoke!(sc; user_callbacks = (callback,));
# @show keys(all_data[0])

# dg(dQdt, Q, nothing, t; increment = false)
# f = joinpath(output_dir, "solution_vs_time.png");
# export_plot(z, all_data, ("ρcT",), f, z_label);

