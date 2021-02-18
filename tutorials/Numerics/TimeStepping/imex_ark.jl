# # [Implicit-Explicit (IMEX) Additively-Partitioned Runge-Kutta Timestepping](@id Single-rate-IMEXARK-Timestepping)

# In this tutorial, we shall explore the use of IMplicit-EXplicit (IMEX) methods
# for the solution of nonautonomous (or non time-invariant) equations.
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
));

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

# To fully demonstrate the advantages of using an IMEX scheme over fully explicit
# schemes, we start here by going over a simple, fully explicit scheme. The
# reader can refer to the [Single-rate Explicit Timestepping tutorial](@ref Single-rate-Explicit-Timestepping)
# for detailes on such schemes. Here we use the the 14-stage LSRK method
# `LSRK144NiegemannDiehlBusch`, which contains the largest stability region of
# the low-storage methods available in `ClimateMachine.jl`.

FT = Float64
timeend = FT(100)

ode_solver = ClimateMachine.ExplicitSolverType(
    solver_method = LSRK144NiegemannDiehlBusch,
);

# In the following example, the timestep calculation is based on the CFL condition
# for horizontally-propogating acoustic waves. We use, ``C = 0.002`` in the
# horizontal, which corresponds to a timestep size of approximately ``1`` second.

C = FT(0.002)
cfl_direction = HorizontalDirection()
run_acousticwave(ode_solver, C, cfl_direction, timeend);

# However, as it is imaginable, for real-world climate processes a time step
# of 1 second would lead to extemely long time-to-solution simulations.
# How can we do better? To be able to take larger time step, we can treat the
# most restrictive wave speeds (vertical acoustic) implicitly rather than
# explicitly. This motivates the use of an IMplicit-EXplicit (IMEX) methods.

# In general, a single step of an ``s``-stage, ``N``-part additive RK method
# (`ARK_N`) is defined by its generalized Butcher tableau:

# ```math
# \begin{align}
#     \begin{array}{c|c|c|c}
#     \boldsymbol{c} &\boldsymbol{A}_{1} & \cdots & \boldsymbol{A}_{N}\\
#     \hline
#     & \boldsymbol{b}_1^T & \cdots & \boldsymbol{b}_N^T\\
#     \hline
#     & \widehat{\boldsymbol{b}}_1^T & \cdots & \widehat{\boldsymbol{b}}_N^T
#     \end{array} =
#     \begin{array}{c|c c c | c | c c c }
#     c_1 & a^{[ 1 ]}_{1,1} & \cdots & a^{[ 1 ]}_{1,s} & \cdots
#     & a^{[ \nu ]}_{1,1} & \cdots & a^{[ \nu ]}_{1,s}\\
#     \vdots & \vdots & \ddots & \vdots & \cdots
#     & \vdots & \ddots & \vdots \\
#     c_s & a^{[ 1 ]}_{s,1} & \cdots & a^{[ 1 ]}_{s,s} & \cdots
#     & a^{[ \nu ]}_{s,1} & \cdots & a^{[ \nu ]}_{s,s}\\
#     \hline
#     & b^{[ 1 ]}_1 & \cdots & b^{[ 1 ]}_s & \cdots
#     & b^{[ \nu ]}_1 & \cdots & b^{[ \nu ]}_s\\
#     \hline
#     & \widehat{b}^{[ 1 ]}_1 & \cdots & \widehat{b}^{[ 1 ]}_s &
#     & \widehat{b}^{[ \nu ]}_1 & \cdots & \widehat{b}^{[ \nu ]}_s
#     \end{array}
# \end{align}
# ```

# and is given by

# ``
# 	\boldsymbol{q}^{n+1} = \boldsymbol{q}^n + \Delta t \left( \underbrace{\sum_{i=1}^{s}}_{\textrm{Stages}} \underbrace{\sum_{\nu=1}^{N}}_{\textrm{Components}} b_i^{[ \nu ]} {\mathcal{T}}^{[ \nu ]}(\boldsymbol{Q}^i)) \right)
# ``

# where ``s`` denotes the stages and ``N`` the components, and where the stage values are given by:
#
# ``
# 	\boldsymbol{Q}^i = \boldsymbol{q}^n + \Delta t \sum_{j=1}^{s} \sum_{\nu = 1}^{N} a_{i,j}^{[ \nu ]}
# 	{\mathcal{T}}^{[ \nu]}(\boldsymbol{Q}^j).
# ``
#
# Similar to standard RK methods, the stage vectors are approximations to the state at each stage
# of the ARK method. Moreover, the temporal coefficients ``c_i`` satisfy a similar
# row-sum condition, holding for all ``\nu = 1, \cdots, N``:

# ``
# 	c_i = \sum_{j=1}^{s} a_{i, j}^{[ \nu ]}, \quad \forall \nu = 1, \cdots, N.
# ``
#
# The Butcher coefficients ``\boldsymbol{c}``, ``\boldsymbol{b}_{\nu}``, ``\boldsymbol{A}_{\nu}``, and ``\widehat{\boldsymbol{b}}_{\nu}``
# are constrained by certain accuracy and stability requirements, which are summarized in
# [Kennedy2001](@cite).

# A common setting is the case ``N = 2``. This gives the typical context for
# Implicit-Explicit (IMEX) splitting methods, where the tendency ``{\mathcal{T}}``
# is assumed to have the decomposition:
#
# ``
# 	\dot{\boldsymbol{q}} = \mathcal{T}(\boldsymbol{q}) \equiv
# 	{\mathcal{T}}_{s}(\boldsymbol{q}) + {\mathcal{T}}_{ns}(\boldsymbol{q}),
# ``
# where the right-hand side has been split into a "stiff" component ``{\mathcal{T}}_{s}``,
# to be treated implicitly, and a non-stiff part ``{\mathcal{T}}_{ns}`` to be treated explicitly.

# Referencing the canonical form introduced in [`Time integration`](@ref
# Time-integration) we have that in this particular forumlation
# ``\mathcal{T}_{ns}(t, \boldsymbol{q}) \equiv \mathcal{G}(t, \boldsymbol{q})`` and
# ``\mathcal{T}_{s}(t, \boldsymbol{q}) \equiv \mathcal{F}(t, \boldsymbol{q})``.

# Two different RK methods are applied to ``{\mathcal{T}}_{s}`` and ``{\mathcal{T}}_{ns}``
# separately, which have been specifically designed and coupled. Examples can be found in
# [Giraldo2013](@cite). The Butcher Tableau for an `ARK_2` method will have the
# form
#
# ```math
# \begin{align}
#     \begin{array}{c|c|c}
#     \boldsymbol{c} &\boldsymbol{A}_E &\boldsymbol{A}_I\\
#     \hline
#     & \boldsymbol{b}_E^T & \boldsymbol{b}_I^T \\
#     \hline
#     & \widehat{\boldsymbol{b}}_E^T & \widehat{\boldsymbol{b}}_I^T
#     \end{array},
# \end{align}
# ```
#
# with
#
# ``
# 	\boldsymbol{A}_O = \left\lbrace a_{i, j}^O \right\rbrace, \quad
# 	\boldsymbol{b}_O = \left\lbrace b_{i}^O \right\rbrace, \quad
# 	\widehat{\boldsymbol{b}}_O = \left\lbrace \widehat{b}_{i}^O \right\rbrace,
# ``
#
# where ``O`` denotes the label (either ``E`` for explicit or ``I`` for implicit).

# For the acoustic wave example used here, we use 4th order polynomials in
# our discontinuous Galerkin approximation, with 6 elements in each horizontal
# direction and 4 elements in the vertical direction, on the cubed-sphere.
# This gives an effective minimanl node-distance (distance between LGL nodes)
# of roughly 203000 m.
# As in the [previous tutorial](@ref Single-rate-Explicit-Timestepping),
# we can determine our ``\Delta t`` by specifying our desired horizontal
# Courant number ``C`` (the timestep calculation is based on the CFL condition
# for horizontally-propogating acoustic waves). In this very simple test case,
# we can use a value of 0.5, which corresponds to a time-step size of
# around 257 seconds. But for this particular example, even higher values
# might work.

timeend = FT(3600)
ode_solver = ClimateMachine.IMEXSolverType(
    solver_method = ARK2GiraldoKellyConstantinescu,
)
C = FT(0.5)
cfl_direction = HorizontalDirection()
run_acousticwave(ode_solver, C, cfl_direction, timeend);

# ## References
# - [Giraldo2013](@cite)
# - [Kennedy2001](@cite)
