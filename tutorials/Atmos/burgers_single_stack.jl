# # Single stack tutorial based on the 3D Burgers + tracer equations

# Equations solved in balance law form:

# ```math
# \begin{align}
# \frac{∂ ρ}{∂ t} =& - ∇ ⋅ (ρ\mathbf{u}) \\
# \frac{∂ ρ\mathbf{u}}{∂ t} =& - ∇ ⋅ (-μ ∇\mathbf{u}) - ∇ ⋅ (ρ\mathbf{u} \mathbf{u}') - γ[ (ρ\mathbf{u}-ρ̄\mathbf{ū}) - (ρ\mathbf{u}-ρ̄\mathbf{ū})⋅ẑ ẑ] \\
# \frac{∂ ρcT}{∂ t} =& - ∇ ⋅ (-α ∇ρcT) - ∇ ⋅ (\mathbf{u} ρcT)
# \end{align}
# ```

# Boundary conditions:
# ```math
# \begin{align}
# z_{\mathrm{min}}: & ρ = 1 \\
# z_{\mathrm{min}}: & ρ\mathbf{u} = \mathbf{0} \\
# z_{\mathrm{min}}: & ρcT = ρc T_{\mathrm{fixed}} \\
# z_{\mathrm{max}}: & ρ = 1 \\
# z_{\mathrm{max}}: & ρ\mathbf{u} = \mathbf{0} \\
# z_{\mathrm{max}}: & -α∇ρcT = 0
# \end{align}
# ```

# where
#  - ``t`` is time
#  - ``ρ`` is the density
#  - ``\mathbf{u}`` is the velocity (vector)
#  - ``\mathbf{ū}`` is the horizontally averaged velocity (vector)
#  - ``μ`` is the dynamic viscosity tensor
#  - ``γ`` is the Rayleigh friction frequency
#  - ``T`` is the temperature
#  - ``α`` is the thermal diffusivity
#  - ``c`` is the heat capacity
#  - ``ρcT`` is the thermal energy

# Solving these equations is broken down into the following steps:
# 1) Preliminary configuration
# 2) PDEs
# 3) Space discretization
# 4) Time discretization
# 5) Solver hooks / callbacks
# 6) Solve
# 7) Post-processing

# # Preliminary configuration

# ## [Loading code](@id Loading-code-burgers)

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using Distributions
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays
using LinearAlgebra: Diagonal

#  - load CLIMAParameters and set up to use it:
using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.BalanceLaws: BalanceLaw
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils

#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
using ClimateMachine.BalanceLaws: BalanceLaw
import ClimateMachine.BalanceLaws:
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
# ClimateMachine.BalanceLaws.BalanceLaw), in this case, the `BurgersEquation`:

Base.@kwdef struct BurgersEquation{FT} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Heat capacity"
    c::FT = 1
    "Vertical dynamic viscosity"
    μv::FT = 1e-4
    "Horizontal dynamic viscosity"
    μh::FT = 1e-2
    "Thermal diffusivity"
    α::FT = 0.01
    "IC Gaussian noise standard deviation"
    σ::FT = 1e-1
    "Rayleigh damping"
    γ::FT = μh / 0.08 / 1e-2 / 1e-2
    "Domain height"
    zmax::FT = 1
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

# Create an instance of the `BurgersEquation`:
m = BurgersEquation{FT}();

# This model dictates the flow control, using [Dynamic Multiple
# Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which
# kernels are executed.

# ## Define the variables

# All of the methods defined in this section were `import`ed in
# [Loading code](@ref Loading-code-burgers) to let us provide
# implementations for our `BurgersEquation` as they will be used
# by the solver.

# Specify auxiliary variables for `BurgersEquation`
vars_state_auxiliary(::BurgersEquation, FT) = @vars(z::FT, T::FT);

# Specify state variables, the variables solved for in the PDEs, for
# `BurgersEquation`
vars_state_conservative(::BurgersEquation, FT) =
    @vars(ρ::FT, ρu::SVector{3, FT}, ρcT::FT);

# Specify state variables whose gradients are needed for `BurgersEquation`
vars_state_gradient(::BurgersEquation, FT) = @vars(u::SVector{3, FT}, ρcT::FT);

# Specify gradient variables for `BurgersEquation`
vars_state_gradient_flux(::BurgersEquation, FT) =
    @vars(μ∇u::SMatrix{3, 3, FT, 9}, α∇ρcT::SVector{3, FT});

# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(
    m::BurgersEquation,
    aux::Vars,
    geom::LocalGeometry,
)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρ`, `state.ρu` and`state.ρcT` are available here because
# we've specified `ρ`, `ρu` and `ρcT` in `vars_state_conservative`
function init_state_conservative!(
    m::BurgersEquation,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    z = aux.z
    ε1 = rand(Normal(0, m.σ))
    ε2 = rand(Normal(0, m.σ))
    state.ρ = 1
    ρu = 1 - 4 * (z - m.zmax / 2)^2 + ε1
    ρv = 1 - 4 * (z - m.zmax / 2)^2 + ε2
    ρw = 0
    state.ρu = SVector(ρu, ρv, ρw)

    state.ρcT = state.ρ * m.c * aux.T
end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.BalanceLaws.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `heat_eq_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::BurgersEquation,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function heat_eq_nodal_update_aux!(
    m::BurgersEquation,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.T = state.ρcT / (state.ρ * m.c)
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT` and `u`. Here, we specify how `ρcT`, `u` are computed. Note that
# `transform.ρcT` and `transform.u` are available here because we've specified `ρcT`
# and `u`in `vars_state_gradient`
function compute_gradient_argument!(
    m::BurgersEquation,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
    transform.u = state.ρu / state.ρ
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.μ∇u` is available here because we've specified `μ∇u` in
#  `vars_state_gradient_flux`
#  - `∇transform.u` is available here because we've specified `u` in
#  `vars_state_gradient`
#  - `diffusive.μ∇u` is built using an anisotropic diffusivity tensor
function compute_gradient_flux!(
    m::BurgersEquation,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇ρcT = m.α * ∇transform.ρcT
    diffusive.μ∇u = Diagonal(SVector(m.μh, m.μh, m.μv)) * ∇transform.u
end;

# Introduce Rayleigh friction towards a target profile as a source.
# Note that:
# - Rayleigh damping is only applied in the horizontal by subtracting
#  the vertical component of momentum from the momentum vector.
function source!(
    m::BurgersEquation{FT},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    args...,
) where {FT}
    ẑ = SVector{3, FT}(0, 0, 1)
    ρ̄ū =
        state.ρ * SVector{3, FT}(
            0.5 - 2 * (aux.z - m.zmax / 2)^2,
            0.5 - 2 * (aux.z - m.zmax / 2)^2,
            0.0,
        )
    ρu_p = state.ρu - ρ̄ū
    source.ρu -= m.γ * (ρu_p - ẑ' * ρu_p * ẑ)
end;

# Compute advective flux.
# Note that:
# - `state.ρu` is available here because we've specified `ρu` in
# `vars_state_conservative`
function flux_first_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    _...,
)
    flux.ρ = state.ρu

    u = state.ρu / state.ρ
    flux.ρu = state.ρu * u'
    flux.ρcT = u * state.ρcT
end;

# Compute diffusive flux (e.g. ``F(μ, \mathbf{u}, t) = -μ∇\mathbf{u}`` in the original PDE).
# Note that:
# - `diffusive.μ∇u` is available here because we've specified `μ∇u` in
# `vars_state_gradient_flux`
function flux_second_order!(
    m::BurgersEquation,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT -= diffusive.α∇ρcT
    flux.ρu -= diffusive.μ∇u
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = μ∇\mathbf{u}``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρ`, `ρu` and `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::BurgersEquation,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # bottom
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
    end
end;

# The boundary conditions for `ρ`, `ρu` and `ρcT` are specified here for
# second-order unknowns
function boundary_state!(
    nf,
    m::BurgersEquation,
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
    if bctype == 1 # bottom
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        state⁺.ρcT = state⁺.ρ * m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0, 0, 0)
        diff⁺.α∇ρcT = -n⁻ * m.flux_top
    end
end;

# # Spatial discretization

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Specify the number of vertical elements
nelem_vert = 20;

# Specify the domain height
zmax = m.zmax;

# Establish a `ClimateMachine` single stack configuration
driver_config = ClimateMachine.SingleStackConfiguration(
    "BurgersEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

# # Time discretization

# Specify simulation time (SI units)
t0 = FT(0)
timeend = FT(10)

# We'll define the time-step based on the [Fourier
# number] and the [Courant number] of the flow
Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / max(m.α, m.μh);
Courant_bound = FT(0.1) * Δ
dt = min(Fourier_bound, Courant_bound)

# # Configure a `ClimateMachine` solver.

# This initializes the state vector and allocates memory for the solution in
# space (`dg` has the model `m`, which describes the PDEs as well as the
# function used for initialization). This additionally initializes the ODE
# solver, by default an explicit Low-Storage
# [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# method.

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# ## Inspect the initial conditions for a single node (e.g. the southwest node)

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(driver_config.grid, z_scale)
state_vars = get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
    i = 1,
    j = 1,
);
aux_vars = get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.dg.state_auxiliary,
    vars_state_auxiliary(m, FT),
    i = 1,
    j = 1,
    exclude = [z_key],
);
all_vars = OrderedDict(state_vars..., aux_vars...);

# Generate plots of initial conditions
export_plot_snapshot(
    z,
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition_T.png"),
    z_label,
);
export_plot_snapshot(
    z,
    all_vars,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_u.png"),
    z_label,
);
export_plot_snapshot(
    z,
    all_vars,
    ("ρu[2]",),
    joinpath(output_dir, "initial_condition_v.png"),
    z_label,
);

# ## Inspect the initial conditions for the horizontal average

# Horizontal statistics of variables
state_vars_var = get_horizontal_variance(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
);

state_vars_avg = get_horizontal_mean(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
);

export_plot_snapshot(
    z,
    state_vars_avg,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_avg_u.png"),
    z_label,
);
export_plot_snapshot(
    z,
    state_vars_var,
    ("ρu[1]",),
    joinpath(output_dir, "initial_condition_variance_u.png"),
    z_label,
);

# ![](initial_condition_avg_u.png)
# ![](initial_condition_variance_u.png)

# # Solver hooks / callbacks

# Define the number of outputs from `t0` to `timeend`
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
dims = OrderedDict(z_key => collect(z));

data_var = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_var[1] = state_vars_var

data_avg = Dict[Dict([k => Dict() for k in 0:n_outputs]...),]
data_avg[1] = state_vars_avg
# The `ClimateMachine`'s time-steppers provide hooks, or callbacks, which
# allow users to inject code to be executed at specified intervals. In this
# callback, the state and aux variables are collected, combined into a single
# `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time) do
    state_vars_var = get_horizontal_variance(
        driver_config.grid,
        solver_config.Q,
        vars_state_conservative(m, FT),
    )
    state_vars_avg = get_horizontal_mean(
        driver_config.grid,
        solver_config.Q,
        vars_state_conservative(m, FT),
    )
    step[1] += 1
    push!(data_var, state_vars_var)
    push!(data_avg, state_vars_avg)
    nothing
end;

# # Solve

# This is the main `ClimateMachine` solver invocation. While users do not have
# access to the time-stepping loop, code may be injected via `user_callbacks`,
# which is a `Tuple` of [`GenericCallbacks`](@ref ClimateMachine.GenericCallbacks).
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in
# `output_dir`.

# Let's plot the horizontal statistics of the solution:
export_plot(
    z,
    data_avg,
    ("ρu[1]"),
    joinpath(output_dir, "solution_vs_time_u.png"),
    z_label,
    xlabel = "Horizontal mean rho*u",
);
export_plot(
    z,
    data_var,
    ("ρu[1]"),
    joinpath(output_dir, "variance_vs_time_u.png"),
    z_label,
    xlabel = "Horizontal variance rho*u",
);
export_plot(
    z,
    data_avg,
    ("ρcT"),
    joinpath(output_dir, "solution_vs_time_T.png"),
    z_label,
    xlabel = "Horizontal mean rho*c*T",
);
export_plot(
    z,
    data_var,
    ("ρu[3]"),
    joinpath(output_dir, "variance_vs_time_w.png"),
    z_label,
    xlabel = "Horizontal variance rho*w",
);
# ![](solution_vs_time_u.png)
# ![](variance_vs_time_u.png)

# The results look as we would expect: they Rayleigh friction damps the
# horizontal velocity to the objective profile and the horizontal
# diffusivity damps the horizontal variance. To run this file, and
# inspect the solution, include this tutorial in the Julia REPL
# with:

# ```julia
# include(joinpath("tutorials", "Atmos", "burgers_single_stack.jl"))
# ```
