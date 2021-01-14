# # [Time integration](@id Time-integration)

# Time integration methods for the numerical solution of Ordinary Differential
# Equations (ODEs), also called timesteppers, can be of different nature and
# flavor (e.g., explicit, semi-implicit, single-stage, multi-stage, single-step,
# multi-step, single-rate, multi-rate, etc). ClimateMachine supports several
# of them. Before showing the different nature of some of these methods, let us
# introduce some common notation.

# A commonly used notation for Initial Value Problems (IVPs) is:

# $$
# \begin{aligned}
#     \frac{\mathrm{d} \boldsymbol{q}}{ \mathrm{d} t} &= \mathcal{T}(t, \boldsymbol{q}),\\
#     \boldsymbol{q}(t_0) &= \boldsymbol{q_0},
# \end{aligned}
# $$

# where $\boldsymbol{q}$ is an unknown function (vector in most of our cases)
# of time $t$, which we would like to approximate, and at the initial time $t_0$
# the corresponding initial value $\boldsymbol{q}_0$ is given.

# The given general formulation, is suitable for single-step explicit schemes.
# Generally, the equation can be represented in the following canonical form:

# $$
# \begin{aligned}
#      {\dot \boldsymbol{q}} + \mathcal{F}(t, \boldsymbol{q}) &= \mathcal{G}(t, \boldsymbol{q}),\\
# \end{aligned}
# $$

# where we have used $\dot \boldsymbol{q} = d \boldsymbol{q} / dt$.
# In both explicit/implicit cases, we refer to the term $\mathcal{G}$
# as the right-hand-side (RHS) or explicit term, and to the spatial terms of
# $\mathcal{F}$ as the left-hand-side or implicit term.
