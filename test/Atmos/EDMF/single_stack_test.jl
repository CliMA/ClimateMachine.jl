# # Eddy Diffusivity- Mass Flux test

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), we'll re-write the equation as:

# "tendency"      = - div("second order flux" + "first order flux") + "non-conservative source"
# \frac{∂ F}{∂ t} = - ∇ ⋅ ( F2 + F1 )                               + S

# where F1 is the flux-componenet that has no gradient term
# where F2 is the flux-componenet that has a  gradient term

# -------------- Subdomains:
# The model has a grid mean (the dycore stats vector),i=1:N updrafts and a single environment subdomain (subscript "0")
# The grid mean is prognostic in first moment and diagnostic in second moment.
# The updrafts are prognostic in first moment and set to zero in second moment.
# The environment is diagnostic in first moment and prognostic in second moment.

# ## Equations solved:
# -------------- First Moment Equations:
#                grid mean
# ``
#     "tendency"           "second order flux"   "first order flux"                 "non-conservative source"
# \frac{∂ ρ}{∂ t}         =                         - ∇ ⋅ (ρu)
# \frac{∂ ρ u}{∂ t}       = - ∇ ⋅ (-ρaK ∇u0       ) - ∇ ⋅ (ρu u' - ρ*MF_{u} )         + S_{surface Friction}
# \frac{∂ ρ e_{int}}{∂ t} = - ∇ ⋅ (-ρaK ∇e_{int,0}) - ∇ ⋅ (u ρ_{int} - ρ*MF_{e_int} ) + S_{microphysics}
# \frac{∂ ρ q_{tot}}{∂ t} = - ∇ ⋅ (-ρaK ∇E_{tot,0}) - ∇ ⋅ (u ρ_{tot} - ρ*MF_{q_tot} ) + S_{microphysics}
# MF_ϕ = \sum{a_i * (w_i-w0)(ϕ_i-ϕ0)}_{i=1:N}
# K is the Eddy_Diffusivity, given as a function of environmental variables
# ``

#                i'th updraft equations (no second order flux)
# ``
#     "tendency"                 "first order flux"    "non-conservative sources"
# \frac{∂ ρa_i}{∂ t}           = - ∇ ⋅ (ρu_i)         + (E_{i0}           - Δ_{i0})
# \frac{∂ ρa_i u_i}{∂ t}       = - ∇ ⋅ (ρu_i u_i')    + (E_{i0}*u_0       - Δ_{i0}*u_i)       + ↑*(ρa_i*b - a_i\frac{∂p^†}{∂z})
# \frac{∂ ρa_i e_{int,i}}{∂ t} = - ∇ ⋅ (ρu*e_{int,i}) + (E_{i0}*e_{int,0} - Δ_{i0}*e_{int,i}) + ρS_{int,i}
# \frac{∂ ρa_i q_{tot,i}}{∂ t} = - ∇ ⋅ (ρu*q_{tot,i}) + (E_{i0}*q_{tot,0} - Δ_{i0}*q_{tot,i}) + ρS_{tot,i}
# b = 0.01*(e_{int,i} - e_{int})/e_{int}
#
#                environment equations first moment
# ``
# a0 = 1-sum{a_i}{i=1:N}
# u0 = (1-sum{a_i*u_i}{i=1:N})/a0
# E_int0 = (1-sum{a_i*E_int_i}{i=1:N})/a0
# q_tot0 = (1-sum{a_i*q_tot_i}{i=1:N})/a0
#
#                environment equations second moment
# ``
#     "tendency"           "second order flux"       "first order flux"  "non-conservative source"
# \frac{∂ ρa_0ϕ'ψ'}{∂ t} =  - ∇ ⋅ (-ρa_0⋅K⋅∇ϕ'ψ')  - ∇ ⋅ (u ρa_0⋅ϕ'ψ')   + 2ρa_0⋅K(∂_z⋅ϕ)(∂_z⋅ψ)  + (E_{i0}*ϕ'ψ' - Δ_{i0}*ϕ'ψ') + ρa_0⋅D_{ϕ'ψ',0} + ρa_0⋅S_{ϕ'ψ',0}
# ``

# --------------------- Ideal gas law and subdomain density
# ``
# T_i, q_l  = saturation adjustment(e_int, q_tot)
# TempShamEquil(e_int,q_tot,p)
# ρ_i = <p>/R_{m,i} * T_i
# b = -g(ρ_i-ρ_h)<ρ>

# where
#  - `t`        is time
#  - `z`        is height
#  - `ρ`        is the density
#  - `u`        is the 3D velocity vector
#  - `e_int`    is the internal energy
#  - `q_tot`    is the total specific humidity
#  - `K`        is the eddy diffusivity
#  - `↑`        is the upwards pointing unit vector
#  - `b`        is the buoyancy
#  - `E_{i0}`   is the entrainment rate from the enviroment into i
#  - `Δ_{i0}`   is the detrainment rate from i to the enviroment
#  - `ϕ'ψ'`     is a shorthand for \overline{ϕ'ψ'}_0 the enviromental covariance of ϕ and ψ
#  - `D`        is a covariance dissipation
#  - `S_{ϕ,i}`  is a source of ϕ in the i'th subdomain
#  - `∂_z`      is the vertical partial derivative

# --------------------- Initial Conditions
# Initial conditions are given for all variables in the grid mean, and subdomain variables assume their grid mean values
# ``
# ------- grid mean:
# ρ = hydrostatic reference state - need to compute that state
# ρu = 0
# ρe_int = convert from input profiles
# ρq_tot = convert from input profiles
# ------- updrafts:
# ρa_i = 0.1/N
# ρau = 0
# ρae_int = a*gm.ρe_int
# ρaq_tot = a*gm.ρq_tot
# ------- environment:
# cld_frac = 0.0
# `ϕ'ψ'` = initial covariance profile
# TKE = initial TKE profile
# ``

# --------------------- Boundary Conditions
#           grid mean
# ``
# surface: ρ =
# z_min: ρu = 0
# z_min: ρe_int = 300*cp ; cp=1000
# z_min: ρq_tot = 0.0
# ``

#           i'th updraft
# ``
# z_min: ρ = 1
# z_min: ρu = 0
# z_min: ρe_int = 302*cp ; cp=1000
# z_min: ρq_tot = 0.0
# ``

# Solving these equations is broken down into the following steps:
# 1) Preliminary configuration
# 2) PDEs
# 3) Space discretization
# 4) Time discretization
# 5) Solver hooks / callbacks
# 6) Solve
# 7) Post-processing

# questions for Charlie
# 1. how do we implement an initial profile rather than an initial value?
#            to this question I would ask how do we identify the vertical level in which we are?
# 2. Can use Thermodynamic state to compute the reference profile of hydrostatic balance and adiabatic
# 3.

# # Preliminary configuration

# ## Loading code

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using Distributions
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays

#  - load CLIMAParameters and set up to use it:

using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d, R_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Thermodynamics
using ClimateMachine.SingleStackUtils
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.SurfaceFluxes
using ClimateMachine.SurfaceFluxes.Nishizawa2018
using ClimateMachine.SurfaceFluxes.Byun1990



#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
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
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
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

include("edmf_model.jl")

# Model parameters can be stored in the particular [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), in this case, a `SingleStack`:

Base.@kwdef struct SingleStack{FT, N, N_quad} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Domain height"
    zmax::FT = 3000
    "subdomain statistics model"
    subdomain_statistics::String = "mean" # needs to be define as a string: "mean", "gaussian quadrature", lognormal quadrature"
    "Virtual temperature at surface (K)"
    T_virt_surf::FT = 300
    "Minimum virtual temperature at the top of the atmosphere (K)"
    T_min_ref::FT = 276
    "Height scale over which virtual temperature drops (m)"
    H_t::FT = 7000
    "EDMF scheme"
    edmf::EDMF{FT, N, N_quad} = EDMF{FT, N, N_quad}()
end

include("edmf_kernels.jl")

N = 2
N_quad = 3
# Create an instance of the `SingleStack`:
m = SingleStack{FT, N, N_quad}();

# This model dictates the flow control, using [Dynamic Multiple
# Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which
# kernels are executed.

# ## Define the variables

# All of the methods defined in this section were `import`ed in # [Loading
# code](@ref) to let us provide implementations for our `SingleStack` as they
# will be used by the solver.

include("dycore_kernels.jl")

# # Spatial discretization

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Specify the number of vertical elements
nelem_vert = 10;

# Specify the domain height
zmax = m.zmax;

# Establish a `ClimateMachine` single stack configuration
driver_config = ClimateMachine.SingleStackConfiguration(
    "SingleStack",
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
# number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 ; # YAIR need to divide by eddy diffusivity here
dt = Fourier_bound

# # Configure a `ClimateMachine` solver.

# This initializes the state vector and allocates memory for the solution in
# space (`dg` has the model `m`, which describes the PDEs as well as the
# function used for initialization). This additionally initializes the ODE
# solver, by default an explicit Low-Storage
# [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# method.

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 1
z_key = "z"
z_label = "z [m]"
z = get_z(driver_config.grid, z_scale)
state_vars = SingleStackUtils.get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.Q,
    vars_state_conservative(m, FT),
);
aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
    driver_config.grid,
    solver_config.dg.state_auxiliary,
    vars_state_auxiliary(m, FT);
    exclude = [z_key]
);
all_vars = OrderedDict(state_vars..., aux_vars...);

export_plot_snapshot(
    z,
    all_vars,
    (
      "ρ",
      "ρe_int",
      "ρq_tot",
      "edmf.updraft[1].ρa",
      "edmf.updraft[1].ρae_int",
      "edmf.updraft[1].ρaq_tot",
    ),
    joinpath(output_dir, "initial_energy.png"),
    z_label,
);
# ![](initial_energy.png)

export_plot_snapshot(
    z,
    all_vars,
    (
     "ρu[1]",
     "ρu[2]",
     "ρu[3]",
     "edmf.updraft[1].ρau[1]",
     "edmf.updraft[1].ρau[2]",
     "edmf.updraft[1].ρau[3]",
     ),
    joinpath(output_dir, "initial_velocity.png"),
    z_label,
);
# ![](initial_velocity.png)

# It matches what we have in `init_state_conservative!(m::SingleStack, ...)`, so
# let's continue.

# # Solver hooks / callbacks

# Define the number of outputs from `t0` to `timeend`
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
dims = OrderedDict(z_key => collect(z));

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"));

all_data = OrderedDict([k => Dict() for k in 0:n_outputs]...)
all_data[0] = deepcopy(all_vars)

# The `ClimateMachine`'s time-steppers provide hooks, or callbacks, which
# allow users to inject code to be executed at specified intervals. In this
# callback, the state and aux variables are collected, combined into a single
# `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    state_vars = SingleStackUtils.get_vars_from_nodal_stack(
        driver_config.grid,
        solver_config.Q,
        vars_state_conservative(m, FT),
    )
    aux_vars = SingleStackUtils.get_vars_from_nodal_stack(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        vars_state_auxiliary(m, FT);
        exclude = [z_key],
    )
    all_vars = OrderedDict(state_vars..., aux_vars...)
    step[1] += 1
    all_data[step[1]] = deepcopy(all_vars)
    nothing
end;

# # Solve

# This is the main `ClimateMachine` solver invocation. While users do not have
# access to the time-stepping loop, code may be injected via `user_callbacks`,
# which is a `Tuple` of [`GenericCallbacks`](@ref).
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in
# `output_dir`. Let's collect them all into a nested dictionary whose keys are
# the output interval. The next level keys are the variable names, and the
# values are the values along the grid:

# all_data = collect_data(output_data, step[1]);

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[0]["T"][:]`
# @show keys(all_data[0])

# Let's plot the solution:

# export_plot(
#     z,
#     all_data,
#     ("ρu[1]","ρu[2]",),
#     joinpath(output_dir, "solution_vs_time.png"),
#     z_label,
# );
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is
# resulting in heat flux that propagates up the domain. To run this file, and
# inspect the solution in `all_data`, include this tutorial in the Julia REPL
# with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
