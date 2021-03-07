# # [Multirate Runge-Kutta Timestepping](@id Multirate-RK-Timestepping)

# In this tutorial, we shall explore the use of Multirate Runge-Kutta
# methods for the solution of nonautonomous (or non time-invariant) equations.
# For our model problem, we shall reuse the acoustic wave test in the GCM
# configuration. See its [code](@ref Acoustic-Wave-Configuration)
# for details on the model and parameters. For the purposes of this tutorial,
# we will only run the experiment for a total of 3600 simulation seconds.
# Details on this test case can be found in Sec. 4.3 of [Giraldo2013](@cite).

using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Numerics",
    "TimeStepping",
    "tutorial_acousticwave_config.jl",
))

FT = Float64;

# The typical context for Multirate splitting methods is given by problems
# in which the tendency ``\mathcal{T}`` is assumed to have single parts that
# operate on different time rates (such as a slow time scale and a fast time scale).
# A general form is given by
#
# ``
# 	\dot{\boldsymbol{q}} = \mathcal{T}(\boldsymbol{q}) \equiv
# 	{\mathcal{T}}_{f}(\boldsymbol{q}) + {\mathcal{T}}_{s}(\boldsymbol{q}),
# ``
#
# where the right-hand side has been split into a "fast" component ``{\mathcal{T}_{f}}``,
# and a "slow" component ``{\mathcal{T}_{s}}``.

# Referencing the canonical form introduced in [Time integration](@ref
# Time-integration), both ``{\mathcal{T}_{f}}`` and ``{\mathcal{T}_{s}}``
# could be discretized either explicitly or implicitly, hence, they could
# belong to either ``\mathcal{F}(t, \boldsymbol{q})`` or ``\mathcal{G}(t, \boldsymbol{q})``
# term.
#
# For a given time-step size ``\Delta t``, the two-rate method in [Schlegel2009](@cite)
# is summarized as the following:
#
# ```math
# \begin{align}
#     \boldsymbol{Q}_1 &= \boldsymbol{q}(t_n), \\
#     \boldsymbol{r}_{i} &= \sum_{j=1}^{i-1}\tilde{a}^O_{ij} {\mathcal{T}_{s}}(\boldsymbol{Q}_{j}), \\
#     \boldsymbol{w}_{i,1} &= \boldsymbol{Q}_{i-1},\\
#     \boldsymbol{w}_{i,k} &= \boldsymbol{w}_{i,k-1} + \Delta t \tilde{c}_i^O \sum_{j=1}^{k-1}\tilde{a}^I_{k,j}
#     \left(\frac{1}{\tilde{c}_i^O}\boldsymbol{r}_i + {\mathcal{T}_{f}}(\boldsymbol{w}_{i,j})\right),\\
#     & \quad\quad i = 2, \cdots, s^O + 1 \text{ and } k = 2, \cdots, s^I + 1,\nonumber \\
#     \boldsymbol{Q}_i &= \boldsymbol{w}_{i,s^I + 1}
# \end{align}
# ```
#
# where the tilde parameters denote increments per RK stage:
#
# ```math
# \begin{align}
#     \tilde{a}_{ij} &= \begin{cases}
#         a_{i,j} - a_{i-1, j} & \text{if } i < s + 1 \\
#         b_j - a_{s,j} & \text{if } i = s + 1
#     \end{cases},\\
#     \tilde{c}_{i} &= \begin{cases}
#         c_{i} - c_{i-1} & \text{if } i < s + 1 \\
#         1 - c_{s} & \text{if } i = s + 1
#     \end{cases},
# \end{align}
# ```
#
# where the coefficients ``a``, ``b``, and ``c`` correspond to the Butcher
# tableau for a given RK method. The superscripts ``O`` and ``I`` denote the
# *outer* (slow) and *inner* (fast) components of the multirate method
# respectively. Thus, tilde coefficients should be associated with the RK
# method indicated by the superscripts. In other words, the RK methods
# for the slow ``{\mathcal{T}_{s}}`` and fast
# ``{\mathcal{T}_{f}}`` components have Butcher tables given by:
#
# ```math
# \begin{align}
#     \begin{array}{c|c}
#     \boldsymbol{c}_{O} &\boldsymbol{A}_{O} \\
#     \hline
#     & \boldsymbol{b}_O^T
#     \end{array}, \quad
#     \begin{array}{c|c}
#     \boldsymbol{c}_{I} &\boldsymbol{A}_{I} \\
#     \hline
#     & \boldsymbol{b}_I^T
#     \end{array},
# \end{align}
# ```
#
# where ``\boldsymbol{A}_O = \lbrace a_{i,j}^O\rbrace``, ``\boldsymbol{b}_O = \lbrace b_i^O \rbrace``, and
# ``c_O = \lbrace c_i^O \rbrace`` (similarly for ``\boldsymbol{A}_I``, ``\boldsymbol{b}_I``, and ``\boldsymbol{c}_I``).
# The method described here is for an explicit RK outer method with ``s`` stages.
# More details can be found in [Schlegel2012](@cite).

# The acoustic wave test case used in this tutorial represents a global-scale
# problem with inertia-gravity waves traveling around the entire planet.
# It has a hydrostatically balanced initial state that is given a pressure
# perturbation.
# This initial pressure perturbation causes an acoustic wave to travel to
# the antipode, coalesce, and return to the initial position. The exact solution
# of this test case is simple in that the (linear) acoustic theory allows one
# to verify the analytic speed of sound based on the thermodynamics variables.
# The initial condition is defined as a hydrostatically balanced atmosphere
# with background (reference) potential temperature.

ode_solver = ClimateMachine.MultirateSolverType(
    splitting_type = ClimateMachine.HEVISplitting(),
    slow_method = LSRK54CarpenterKennedy,
    fast_method = ARK2GiraldoKellyConstantinescu,
    implicit_solver_adjustable = true,
    timestep_ratio = 100,
)

timeend = FT(3600)
CFL = FT(5)
cfl_direction = HorizontalDirection()
run_acousticwave(ode_solver, CFL, cfl_direction, timeend);

# The interested reader can explore the combination of different slow and
# fast methods for Multirate solvers, consulting the ones available in
# `ClimateMachine.jl`, such as the
# [`Low-Storage-Runge-Kutta-methods`](@ref ClimateMachine.ODESolvers.LowStorageRungeKutta2N),
# [`Strong-Stability-Preserving-RungeKutta-methods`](@ref ClimateMachine.ODESolvers.StrongStabilityPreservingRungeKutta),
# and [`Additive-Runge-Kutta-methods`](@ref ClimateMachine.ODESolvers.AdditiveRungeKutta).

# ## References
# - [Giraldo2013](@cite)
# - [Schlegel2009](@cite)
# - [Schlegel2012](@cite)
