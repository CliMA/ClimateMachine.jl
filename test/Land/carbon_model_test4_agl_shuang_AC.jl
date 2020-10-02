# # Heat equation tutorial

# In this tutorial, we'll be solving the [heat
# equation](https://en.wikipedia.org/wiki/Heat_equation):

# ``
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (-α ∇ρcT) = 0
# ``

# where
#  - `t` is time
#  - `α` is the thermal diffusivity
#  - `T` is the temperature
#  - `ρ` is the density
#  - `c` is the heat capacity
#  - `ρcT` is the thermal energy

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref
# ClimateMachine.BalanceLaws.BalanceLaw), we'll re-write the equation as:

# ``
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (F(α, ρcT, t)) = 0
# ``

# where
#  - ``F(α, ρcT, t) = -α ∇ρcT`` is the second-order flux

# with boundary conditions
#  - Fixed temperature ``T_{surface}`` at ``z_{min}`` (non-zero Dirichlet)
#  - No thermal flux at ``z_{min}`` (zero Neumann)

# Solving these equations is broken down into the following steps:
# 1) Preliminary configuration
# 2) PDEs
# 3) Space discretization
# 4) Time discretization / solver
# 5) Solver hooks / callbacks
# 6) Solve
# 7) Post-processing

# # Preliminary configuration

# ## [Loading code](@id Loading-code-heat)

# First, we'll load our pre-requisites:
#  - load external packages:
using MPI
using OrderedCollections
using Plots
using StaticArrays
using OrdinaryDiffEq
using DiffEqBase

#  - load CLIMAParameters and set up to use it:

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
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

import ClimateMachine.DGMethods: calculate_dt

# ## Initialization

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize ClimateMachine for CPU.
ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));

# Load some helper functions for plotting
include(joinpath(clima_dir, "docs", "plothelpers.jl"));

# # Define the set of Partial Differential Equations (PDEs)

# ## Define the model

# Model parameters can be stored in the particular [`BalanceLaw`](@ref
# ClimateMachine.BalanceLaws.BalanceLaw), in this case, a `HeatModel`:

# add in the 3 other Bs (NPP), and a matrix containing all the other transf coefs
# replace ks by tau, tau=1/k
Base.@kwdef struct CarbonModel{FT, APS} <: BalanceLaw
    "Parameters"
    param_set::APS
    "Initial Fine litter (g carbon/m^2)"
    FL_init::FT = 120
    "Initial Structural litter (g carbon/m^2)"
    SL_init::FT = 300
    "Initial fast SOC (soil) (g carbon/m^2)"
    FSOC_init::FT = 350.0
    "Initial slow SOC (soil) (g carbon/m^2)"
    SSOC_init::FT = 86000.0
    "Initial passive SOC (soil) (g carbon/m^2)"
    PSOC_init::FT = 71000.0
    "Net primary production (g carbon/m^2/yr)"
    NPP::FT = 295.15#/(365*3600/100)
    "k_1 (g C/yr)"
    k_1::FT = (1/0.2)#/(365*3600/100)#0.28 0.32
    "k_2 (g C/yr)"
    k_2::FT = (1/3.9)#/(365*3600/100)#15.0 3.9
    "k_3 (g C/yr)"
    k_3::FT = (1/0.42)#/(365*3600/100)#0.48 0.42
    "k_4 (g C/yr)"
    k_4::FT = (1/70.5)#/(365*3600/100)#500  70.5
    "k_5 (g C/yr)"
    k_5::FT = (1/1200)#/(365*3600/100)#7200
    "carbon flow from fine litter to fast SOC (1/yr)"
    a_31::FT = 0.55
    "carbon flow from structural litter to fast SOC (1/yr)"
    a_32::FT = 0.275
    "carbon flow from structural litter to slow SOC (1/yr)"
    a_42::FT = 0.275
    "carbon flow from fast SOC to slow SOC (1/yr)"
    a_43::FT = 0.3
    "carbon flow from passive SOC to slow SOC (1/yr)"
    a_53::FT = 0.1
    "carbon flow from slow to fast SOC (1/yr)"
    a_34::FT = 0.5
    "carbon flow from slow SOC to passive SOC (1/yr)"
    a_54::FT = 0.2
    "carbon flow from passive SOC to fast SOC (1/yr)"
    a_35::FT = 0.45
    "Diffusivity (units?)"
    D::FT = 0.001
end

# what is physical process that happens most quickly, and make sure we resolve this
# start with yearly time step. daily time step?  eventually we will want hourly timesteps

# Create an instance of the `HeatModel`:
m = CarbonModel{FT, typeof(param_set)}(; param_set = param_set);

# This model dictates the flow control, using [Dynamic Multiple
# Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which
# kernels are executed.

# ## Define the variables

# All of the methods defined in this section were `import`ed in
# [Loading code](@ref Loading-code-heat) to let us provide
# implementations for our `HeatModel` as they will be used by
# the solver.

# Specify auxiliary variables for `CarbonModel`
vars_state(::CarbonModel, ::Auxiliary, FT) = @vars(z::FT, FLk_1::FT, SLk_2::FT, FSOCk_3::FT, SSOCk_4::FT, PSOCk_5::FT);

# Specify state variables, the variables solved for in the PDEs, for
# `CarbonModel`
vars_state(::CarbonModel, ::Prognostic, FT) = @vars(FL::FT, SL::FT, FSOC::FT, SSOC::FT, PSOC::FT);

# Specify state variables whose gradients are needed for `CarbonModel`
vars_state(::CarbonModel, ::Gradient, FT) = @vars();

# Specify gradient variables for `CarbonModel`
vars_stat(::CarbonModel, ::GradientFlux, FT) = @vars();

# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_prognostic!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state` given `Auxiliary`
# in `vars_state`

# function of state as helper function, we dont want this at every time step, fewer aux vars is better
function carbon_eq_nodal_init_state_auxiliary!(
    m::CarbonModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    aux.FLk_1 = m.FL_init * m.k_1 #... why was this not in aux? rate at which carbon leaves pool B
    aux.SLk_2 = m.SL_init * m.k_2
    aux.FSOCk_3 = m.FSOC_init * m.k_3
    aux.SSOCk_4 = m.SSOC_init * m.k_4
    aux.PSOCk_5 = m.PSOC_init * m.k_5
end;

function init_state_auxiliary!(
    m::CarbonModel,
    state_auxiliary::MPIStateArray,
    grid,
)
    nodal_init_state_auxiliary!(
        m,
        carbon_eq_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
    )
end

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in
# `vars_state` given `Prognostic`
function init_state_prognostic!(
    m::CarbonModel,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    state.FL = m.FL_init
    state.SL = m.SL_init
    state.FSOC = m.FSOC_init
    state.SSOC = m.SSOC_init
    state.PSOC = m.PSOC_init
end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.BalanceLaws.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `heat_eq_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::CarbonModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(Carbon_eq_nodal_update_aux!, dg, m, Q, t, elems)
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state` given `Auxiliary`
function Carbon_eq_nodal_update_aux!(
    m::CarbonModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.FLk_1 = state.FL * m.k_1 #... why was this not in aux? rate at which carbon leaves pool B
    aux.SLk_2 = state.SL * m.k_2
    aux.FSOCk_3 = state.FSOC * m.k_3
    aux.SSOCk_4 = state.SSOC * m.k_4
    aux.PSOCk_5 = state.PSOC * m.k_5
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state` given `Gradient`
function compute_gradient_argument!(
    m::CarbonModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
#  `vars_state` given `Gradient`
#  - `∇transform.ρcT` is available here because we've specified `ρcT`  in
#  `vars_state` given `Gradient`
function compute_gradient_flux!(
    m::CarbonModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::CarbonModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    source.FL = m.NPP*0.75  - aux.FLk_1# net rate of incoming carbon into B - outgoing carbon into S
    source.SL = m.NPP*0.25 - aux.SLk_2 # net rate of incoming carbon into S - outgoing carbon to ...?
    source.FSOC =  m.a_35*aux.PSOCk_5 + m.a_34*aux.SSOCk_4 + m.a_32*aux.SLk_2 + m.a_31*aux.FLk_1 - aux.FSOCk_3 # net rate of incoming carbon into B - outgoing carbon into S
    source.SSOC =  m.a_42*aux.SLk_2 + m.a_43*aux.FSOCk_3 - aux.SSOCk_4 # net rate of incoming carbon into S - outgoing carbon to ...?
    source.PSOC =  m.a_53*aux.FSOCk_3 + m.a_54*aux.SSOCk_4 - aux.PSOCk_5 # net rate of incoming carbon into B - outgoing carbon into S
end;

#aux.FLk_1 = state.FL * m.k_1 #... why was this not in aux? rate at which carbon leaves pool B
#aux.SLk_2 = state.SL * m.k_2
#aux.FSOCk_3 = state.FSOC * m.k_3
#aux.SSOCk_4 = state.SSOC * m.k_4
#aux.PSOCk_5 = state.PSOC * m.k_5

function flux_first_order!(m::CarbonModel, _...) end;

# Compute diffusive flux (``F(α, ρcT, t) = -α ∇ρcT`` in the original PDE).
# Note that:
# - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
# `vars_state` given `GradientFlux`
function flux_second_order!(
    m::CarbonModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(F)`` where ``F = -α∇ρcT``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::CarbonModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
end;

# The boundary conditions for `ρcT` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::CarbonModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
end;

# # Spatial discretization

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Specify the number of vertical elements
nelem_vert = 10;

# Specify the domain height
zmax = FT(1);

# Establish a `ClimateMachine` single stack configuration
driver_config = ClimateMachine.SingleStackConfiguration(
    "HeatEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

# # Time discretization / solver

# Specify simulation time (SI units)
t0 = FT(0)
timeend = FT(10000)
#dt = FT(10)

# In this section, we initialize the state vector and allocate memory for
# the solution in space (`dg` has the model `m`, which describes the PDEs
# as well as the function used for initialization). `SolverConfiguration`
# initializes the ODE solver, by default an explicit Low-Storage
# [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# method. In this tutorial, we prescribe an option for an implicit
# `Kvaerno3` method.

# First, let's define how the time-step is computed, based on the
# [Fourier number](https://en.wikipedia.org/wiki/Fourier_number)
# (i.e., diffusive Courant number) is defined. Because
# the `HeatModel` is a custom model, we must define how both are computed.
# First, we must define our own implementation of `DGMethods.calculate_dt`,
# (which we imported):
function calculate_dt(dg, model::CarbonModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    return Courant_number / CFL
end

# Next, we'll define our implementation of `diffusive_courant`:
function diffusive_courant(
    m::CarbonModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
    return Δt * m.D / (Δx * Δx)
end

# Finally, we initialize the state vector and solver
# configuration based on the given Fourier number.
# Note that, we can use a much larger Fourier number
# for implicit solvers as compared to explicit solvers.
use_implicit_solver = false
if use_implicit_solver
    given_Fourier = FT(30)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        ode_solver_type = ImplicitSolverType(OrdinaryDiffEq.Kvaerno3(
            autodiff = false,
            linsolve = LinSolveGMRES(),
        )),
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
else
    given_Fourier = FT(0.7)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
end;


grid = solver_config.dg.grid;
Q = solver_config.Q;
aux = solver_config.dg.state_auxiliary;

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100; # convert from meters to cm
z_key = "z";
z_label = "z [cm]";
z = get_z(grid, z_scale);

# Create an array to store the solution:
all_data = Dict[dict_of_nodal_states(solver_config, [z_key])]  # store initial condition at ``t=0``
time_data = FT[0]                                      # store time data

export_plot(
    z,
    all_data,
    ("FL","SL","FSOC","SSOC","PSOC"),
    joinpath(output_dir, "initial_condition.png"),
    xlabel = "g carbon/m^2",
    ylabel = z_label,
    time_data = time_data,
);
# ![](initial_condition.png)

# It matches what we have in `init_state_prognostic!(m::HeatModel, ...)`, so
# let's continue.

# # Solver hooks / callbacks

# Define the number of outputs from `t0` to `timeend`
const n_outputs = 20;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# The `ClimateMachine`'s time-steppers provide hooks, or callbacks, which
# allow users to inject code to be executed at specified intervals. In this
# callback, a dictionary of prognostic and auxiliary states are appended to
# `all_data` for time the callback is executed. In addition, time is collected
# and appended to `time_data`.
callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    push!(all_data, dict_of_nodal_states(solver_config, [z_key]))
    push!(time_data, gettime(solver_config.solver))
    nothing
end;

# # Solve

# This is the main `ClimateMachine` solver invocation. While users do not have
# access to the time-stepping loop, code may be injected via `user_callbacks`,
# which is a `Tuple` of callbacks in [`GenericCallbacks`](@ref ClimateMachine.GenericCallbacks).
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,));

# Append result at the end of the last time step:
push!(all_data, dict_of_nodal_states(solver_config, [z_key]));
push!(time_data, gettime(solver_config.solver));

# # Post-processing

# Our solution is stored in the array of dictionaries `all_data` whose keys are
# the output interval. The next level keys are the variable names, and the
# values are the values along the grid:

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[1]["T"][:]`
@show keys(all_data[1])

# Let's plot the solution:

FL_vs_t = [all_data[i]["FL"][1] for i in keys(all_data)]
SL_vs_t = [all_data[i]["SL"][1] for i in keys(all_data)]
FSOC_vs_t = [all_data[i]["FSOC"][1] for i in keys(all_data)]
SSOC_vs_t = [all_data[i]["SSOC"][1] for i in keys(all_data)]
PSOC_vs_t = [all_data[i]["PSOC"][1] for i in keys(all_data)]
plot(time_data, FL_vs_t, label = "FL")
plot!(time_data, SL_vs_t, label = "SL")
plot!(time_data, FSOC_vs_t, label = "FSOC")
plot!(ylims=(0,500))

plot(time_data, SSOC_vs_t, label = "SSOC")
plot!(time_data, PSOC_vs_t, label = "PSOC")
plot!(ylims=(70000,90000))
savefig(joinpath(output_dir, "sol_vs_time.png"))
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is
# resulting in heat flux that propagates up the domain. To run this file, and
# inspect the solution in `all_data`, include this tutorial in the Julia REPL
# with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
