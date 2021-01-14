# # [Single-rate Explicit Timestepping](@id Single-rate-Explicit-Timestepping)

# In this tutorial, we shall explore the use of explicit Runge-Kutta
# methods for the solution of nonautonomous (or non time-invariant) equations.
# For our model problem, we shall reuse the rising thermal bubble
# tutorial. See its [tutorial page](@ref Rising-Thermal-Bubble-Configuration)
# for details on the model and parameters. For the purposes of this tutorial,
# we will only run the experiment for a total of 100 simulation seconds.

include(joinpath(
    @__DIR__,
    "../../../../../",
    "tutorials/Numerics/TimeStepping/tutorial_risingbubble_config.jl",
))

FT = Float64
timeend = FT(100)

# After discretizing the spatial terms in the equation, the semi-discretization
# of the governing equations have the form:

# $$
# \begin{aligned}
#     \frac{\mathrm{d} \boldsymbol{q}}{ \mathrm{d} t} &= M^{-1}\left(M S +
#     D^{T} M (F^{adv} + F^{visc}) + \sum_{f=1}^{N_f} L^T M_f(\widehat{F}^{adv} + \widehat{F}^{visc})
#     \right) \equiv \mathcal{T}(\boldsymbol{q}).
# \end{aligned}
# $$

# Referencing the canonical form introduced in [`Time integration`](@ref
# Time-integration) we have that in any explicit
# formulation $\mathcal{F}(t, \boldsymbol{q}) \equiv 0$ and, in this particular
# forumlation $\mathcal{T}(t, \boldsymbol{q}) \equiv \mathcal{G}(t, \boldsymbol{q})$.

# The time-step restriction for an explicit method must satisfy the stable
# [Courant number](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)
# for the specific time-integrator and must be selected from the following
# constraints
#
# $$
# \begin{aligned}
#     \Delta t_{\mathrm{explicit}} = min \left( \frac{C \Delta x_i}{u_i + a}, \frac{C \Delta x_i^2}{\nu} \right)
# \end{aligned}
# $$
#
# where $C$ is the stable Courant number, $u_i$ denotes the velocity components,
# $a$ the speed of sound, $\Delta x_i$ the grid spacing (non-uniform in case of
# spectral element methods) along the direction $(x_1,x_2,x_3)$, and $\nu$ the
# kinematic viscosity. The first term on the right is the time-step condition
# due to the non-dissipative components while the second term to the dissipation.
# For explicit time-integrators, we have to find the minimum time-step that
# satisfies this condition along all three spatial directions.
#
# ## Low-storage Runge-Kutta methods
#
# A single step of an ``s``-stage Runge-Kutta (RK) method for
# solving the resulting ODE problem in [eq:foo] and can be
# expressed as the following:
#
# $$
# \begin{aligned}
# 	\boldsymbol{q}^{n+1} = \boldsymbol{q}^n + \Delta t \sum_{i=1}^{s} b_i \mathcal{T}(\boldsymbol{Q}^i),
# \end{aligned}
# $$
#
# where $\boldsymbol{\mathcal{T}}(\boldsymbol{Q}^i)$ is the evaluation of the
# right-hand side tendency at the stage value $\boldsymbol{Q}^i$, defined at
# each stage of the RK method:
#
# $$
# \begin{aligned}
# 	\boldsymbol{Q}^i = \boldsymbol{q}^{n} +
#     \Delta t \sum_{j=1}^{s} a_{i,j}
#     \mathcal{T}(\boldsymbol{Q^j}).
# \end{aligned}
# $$
#
# The first stage is initialized using the field at the previous time-step:
# $\boldsymbol{Q}^{1} = \boldsymbol{q}^n$.
#
# In the above expressions, we define $\boldsymbol{A} = \lbrace a_{i,j} \rbrace \in \mathbb{R}^{s\times s}$, $\boldsymbol{b} = \lbrace b_i \rbrace \in \mathbb{R}^s$, and $\boldsymbol{c} = \lbrace c_i \rbrace \in \mathbb{R}^s$ as the characteristic coefficients of a given RK method. This means we can associate any RK method with its so-called *Butcher tableau*:
#
# $$
# \begin{aligned}
#     \begin{array}{c|c}
#         \boldsymbol{c} &\boldsymbol{A}\\
#         \hline
#         & \boldsymbol{b}^T
#         \end{array} =
#         \begin{array}{c|c c c c}
#         c_1 & a_{1,1} & a_{1,2} & \cdots & a_{1,s}\\
#         c_2 & a_{2,1} & a_{2,2} & \cdots & a_{2,s}\\
#         \vdots & \vdots & \vdots & \ddots & \vdots\\
#         c_s & a_{s,1} & a_{s,2} & \cdots & a_{s,s}\\
#         \hline
#         & b_1 & b_2 & \cdots & b_s
#     \end{array}.
# \end{aligned}
# $$
#
# The vector $\boldsymbol{c}$ is often called the *consistency vector*,
# and is typically subject to the row-sum condition:
#
# $$
# c_i = \sum_{j=1}^{s} a_{i,j}, \quad \forall i = 1, \cdots, s.
# $$
#
# This simplifies the order conditions for higher-order RK methods.
# For more information on general RK methods, we refer the interested reader
# to \cite[Ch. 5.2]{atkinson2011numerical}.
#
# ## [Low-storage Runge-Kutta methods](@id lsrk)
# For our first example, we shall run a simple rising bubble
# experiment for 100 seconds. [...]
# ClimateMachine.jl contains the following low-storage methods:
#   - Forward Euler (`LSRKEulerMethod`)
#   - A 5-stage 4th-order Runge-Kutta method of Carpenter and Kennedy (`LSRK54CarpenterKennedy`)
#   - A 14-stage 4th-order Runge-Kutta method developed by Niegemann, Diehl, and Busch (`LSRK144NiegemannDiehlBusch`).
#
# To start, let's try using the 5-stage method: `LSRK54CarpenterKennedy`.

# As is the case for all explicit methods, we are limited by the fastest
# propogating waves described by our governing equations. In our case,
# these are the acoustic wave speeds (approximately 343 m/s).
# For the rising bubble example used here, we use 4th order polynomials in
# our discontinuous Galerkin approximation, with a domain resolution of
# 125 meters in each spatial direction. This gives an effective
# minimanl node-distance (distance between LGL nodes) of 86 meters
# over the entire mesh. Using the equation for the explciit time-step above,
# we can determine our $\Delta t$ by specifying our desired Courant number $C$.
# In our case, a heuristically determined value of 0.4 is used.

ode_solver =
    ClimateMachine.ExplicitSolverType(solver_method = LSRK54CarpenterKennedy)
C = FT(0.4)
run_simulation(ode_solver, C, timeend)

# What if we wish to take a larger timestep size? To do this, we can
# try to increase the target Courant number, say $C = 1.7$, and
# re-run the simulation:
C = FT(1.7)

# Oh-no, it breaks! What has happened in this case is our simulation
# has gone unstable and crashed. This occurs when the time-step
# _exceeds_ the maximal stable time-step size of the method. For
# the 5-stage method, one can typically get away with using time-step
# sizes corresponding to a Courant number of $C \approx 0.4$ but
# typically not much larger. In contrast, we can use an LSRK method with
# a larger stability region. Let's try using the 14-stage method instead.

ode_solver = ClimateMachine.ExplicitSolverType(
    solver_method = LSRK144NiegemannDiehlBusch,
)
run_simulation(ode_solver, C, timeend)

# And it completes. Currently, the 14-stage LSRK method `LSRK144NiegemannDiehlBusch`
# contains the largest stability region of the low-storage methods.

# # Strong Stability Preserving Runge--Kutta methods

# Just as with the LSRK methods, the SSPRK methods are self-starting,
# with $\boldsymbol{Q}^{1} = \boldsymbol{q}^n$, and stage-values are of the form

# $$
# \begin{aligned}
#     \boldsymbol{Q}^{i+1} = a_{i,1} \boldsymbol{q}^n
#     + a_{i,2} \boldsymbol{Q}^{i}
#     + \Delta t b_i\mathcal{T}(\boldsymbol{Q}^{i})
# \end{aligned}
# $$
#
# with the value at the next step being the $(N+1)$-th stage value $\boldsymbol{q}^{n+1} = \boldsymbol{Q}^{(N+1)}$. This allows the updates to be performed with only three copies of the state vector (storing $\boldsymbol{q}^n$, $\boldsymbol{Q}^{i}$ and $\mathcal{T}(\boldsymbol{Q}^{i})$).

# [Reference literature for the theoretical construction of the SSPRK methods]

# ## [Strong-stability-preserving Runge-Kutta methods](@id ssprk)

ode_solver = ClimateMachine.ExplicitSolverType(solver_method = SSPRK33ShuOsher)
C = FT(0.2)
run_simulation(ode_solver, C, timeend)
