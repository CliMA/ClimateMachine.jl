# # [Multirate Infinitesimal Step (MIS) Timestepping](@id MIS-Timestepping)

# In this tutorial, we shall explore the use of explicit Runge-Kutta
# methods for the solution of nonautonomous (or non time-invariant) equations.
# For our model problem, we shall reuse the rising thermal bubble
# tutorial. See its [tutorial page](@ref Rising-Thermal-Bubble-Configuration)
# for details on the model and parameters. For the purposes of this tutorial,
# we will only run the experiment for a total of 100 simulation seconds.

using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(
    clima_dir,
    "tutorials",
    "Numerics",
    "TimeStepping",
    "tutorial_risingbubble_config.jl",
))

FT = Float64;

# In this tutorial, we shall explore the use of Multirate Infinitesimal Step
# (MIS) methods for the solution of nonautonomous (or non time-invariant) equations.
# For our model problem, we shall reuse the acoustic wave test in the GCM
# configuration. See its [code](@ref Acoustic-Wave-Configuration)
# for details on the model and parameters. For the purposes of this tutorial,
# we will only run the experiment for a total of 1800 simulation seconds.
# Details on this test case can be found in Sec. 4.3 of [Giraldo2013](@cite).

# Referencing the formulation introduced in the previous
# [Multirate RK methods tutorial](@ref Multirate-RK-Timestepping), we can
# describe Multirate Infinitesimal Step (MIS) methods by

# ```math
# \begin{align}
# v_i (0)
#   &= q^n + \sum_{j=1}^{i-1} \alpha_{ij} (Q^{(j)} - q^n) \\
# \frac{dv_i}{d\tau}
#   &= \sum_{j=1}^{i-1} \frac{\gamma_{ij}}{d_i \Delta t} (Q^{(j)} - q^n)
#     + \sum_{j=1}^i \frac{\beta_{ij}}{d_i} \mathcal{T}_S (Q^{(j)}, t + \Delta t c_i)
#     + \mathcal{T}_F(v_i, t^n +  \Delta t \tilde c_i + \frac{c_i - \tilde c_i}{d_i} \tau),
# \quad \tau \in [0, \Delta t d_i] \\
# Q^{(i)} &= v_i(\Delta t d_i),
# \end{align}
# ```
#
# where we have used the the stage values ``Q^{(i)} = v_i(\tau_i)`` as the
# solution to the _inner_ ODE problem, ``{\mathcal{T}_{s}}``
# for the slow component, and ``{\mathcal{T}_{f}}` for the fast
# one, as in the [Multirate RK methods tutorial](@ref Multirate-RK-Timestepping).
#
# The method is defined in terms of the lower-triangular matrices ``\alpha``,
# ``\beta`` and ``\gamma``, with ``d_i = \sum_j \beta_{ij}``,
# ``c_i = (I - \alpha - \gamma)^{-1} d`` and ``\tilde c = \alpha c``.
# More details can be found in [WenschKnothGalant2009](@cite) and
# [KnothWensch2014](@cite).

ode_solver = ClimateMachine.MISSolverType(;
    mis_method = MIS2,
    fast_method = LSRK144NiegemannDiehlBusch,
    nsubsteps = (40,),
)

timeend = FT(500)
C = FT(20)
run_simulation(ode_solver, C, timeend);

# The reader can compare the Courant number used in this example, with the
# adopted in the
# [single-rate explicit timestepping tutorial page](@ref Single-rate-Explicit-Timestepping)
# in which we use the same scheme as the fast method employed in this case,
# and notice that with this MIS method we are able to take a much larger
# Courant number.

# ## References
# - [WenschKnothGalant2009](@cite)
# - [KnothWensch2014](@cite)
