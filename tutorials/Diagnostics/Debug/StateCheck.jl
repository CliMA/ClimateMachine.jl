# # State debug statistics
#
# This page shows how to use the `StateCheck` functions to get basic
# statistics for nodal values of fields held in ClimateMachine `MPIStateArray`
# data structures. The `StateCheck` functions can be used to
#
# 1. Generate statistics on `MPIStateArray` holding the state of a ClimateMachine experiment.
#
#    and to
#
# 2. Compare against saved reference statistics from ClimateMachine `MPIStateArray`
#    variables. This can enable simple automated regression test checks for
#    detecting unexpected changes introduced into numerical experiments
#    by code updates.
#
# These two cases are shown below:


# ## 1. Generating statistics for a set of MPIStateArrays
#
# Here we create a callback that can generate statistics for an arbitrary
# set of the MPIStateArray type variables of the sort that hold persistent state for
# ClimateMachine models. We then invoke the call back to show the statistics.
#
# In regular use the `MPIStateArray` variables will come from model configurations.
# Here we create a dummy set of `MPIStateArray` variables for use in stand alone
# examples.

# ### Create a dummy set of MPIStateArrays
#
# First we set up two `MPIStateArray` variables. This need a few packages to be in placeT,
# and utilizes some utility functions to create the array and add named
# persistent state variables.
# This is usually handled automatically as part of model definition in regular
# ClimateMachine activity.
# Calling `ClimateMachine.init()` includes initializing GPU CUDA and MPI parallel
# processing options that match the hardware/software system in use.

# Set up a basic environment
using MPI
using StaticArrays
using Random
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.StateCheck

ClimateMachine.init()
FT = Float64

# Define some dummy vector and tensor abstract variables with associated types
# and dimensions
F1 = @vars begin
    ν∇u::SMatrix{3, 2, FT, 6}
    κ∇θ::SVector{3, FT}
end
F2 = @vars begin
    u::SVector{2, FT}
    θ::SVector{1, FT}
end
nothing # hide

# Create `MPIStateArray` variables with arrays to hold elements of the
# vectors and tensors
Q1 = MPIStateArray{Float32, F1}(
    MPI.COMM_WORLD,
    ClimateMachine.array_type(),
    4,
    9,
    8,
)
Q2 = MPIStateArray{Float64, F2}(
    MPI.COMM_WORLD,
    ClimateMachine.array_type(),
    4,
    6,
    8,
)
nothing # hide

# ### Create a call-back
#
# Now we can create a `StateCheck` call-back, _cb_, tied to the `MPIStateArray`
# variables _Q1_ and _Q2_. Each `MPIStateArray` in the array
# of `MPIStateArray` variables tracked is paired with a label
# to identify it. The call-back is also given a frequency (in time step numbers) and
# precision for printing summary tables.
cb = ClimateMachine.StateCheck.sccreate(
    [(Q1, "My gradients"), (Q2, "My fields")],
    1;
    prec = 15,
)
GenericCallbacks.init!(cb, nothing, nothing, nothing, nothing)
nothing # hide

# ### Invoke the call-back
#
# The call-back is of type `ClimateMachine.GenericCallbacks.EveryXSimulationSteps`
# and in regular use is designed to be passed to a ClimateMachine timestepping
# solver e.g.
typeof(cb)

# Here, for demonstration purposes, we can invoke
# the call-back after simply initializing the `MPIStateArray` fields to a random
# set of values e.g.
Q1.data .= rand(MersenneTwister(0), Float32, size(Q1.data))
Q2.data .= rand(MersenneTwister(0), Float64, size(Q2.data))
GenericCallbacks.call!(cb, nothing, nothing, nothing, nothing)

# ## 2. Comparing to reference values

# ### Generate arrays of reference values
#
# StateCheck functions can generate text that can be used to set the value of stored
# arrays that can be used in a reference test for subsequent regression testing. This
# involves 3 steps.
#
# **Step 1.** First a reference array setting program code is generated from the latest
# state of a given callback e.g.

ClimateMachine.StateCheck.scprintref(cb)

# **Step 2.** Next the array setting program code is executed (see below). At this stage the _parr[]_ array
# context may be hand edited. The parr[] array sets a target number of decimal places for
# matching against reference values in _varr[]_. For different experiments and different fields
# the degree of precision that constitutes failing a regression test may vary. Choosing the
# _parr[]_ values requires some sense as to the stability of the particular numerical
# and physical scenario an experiment represents. In the example below some precision
# settings have been hand edited from the default of 16 to illustrate the process.

#! format: off
varr = [
 [ "My gradients", "ν∇u[1]",  1.34348869323730468750e-04,  9.84732866287231445313e-01,  5.23545503616333007813e-01,  3.08209930764271777814e-01 ],
 [ "My gradients", "ν∇u[2]",  1.16317868232727050781e-01,  9.92088317871093750000e-01,  4.83800649642944335938e-01,  2.83350456014221541157e-01 ],
 [ "My gradients", "ν∇u[3]",  1.05845928192138671875e-03,  9.51775908470153808594e-01,  4.65474426746368408203e-01,  2.73615551085745090099e-01 ],
 [ "My gradients", "ν∇u[4]",  5.97668886184692382813e-02,  9.68048095703125000000e-01,  5.42618036270141601563e-01,  2.81570862027933854765e-01 ],
 [ "My gradients", "ν∇u[5]",  8.31030607223510742188e-02,  9.35931921005249023438e-01,  5.05405902862548828125e-01,  2.46073509972619536290e-01 ],
 [ "My gradients", "ν∇u[6]",  3.09681892395019531250e-02,  9.98341441154479980469e-01,  4.54375565052032470703e-01,  3.09461067853178561915e-01 ],
 [ "My gradients", "κ∇θ[1]",  8.47448110580444335938e-02,  9.94180679321289062500e-01,  5.27157366275787353516e-01,  2.92455951648181833313e-01 ],
 [ "My gradients", "κ∇θ[2]",  1.20514631271362304688e-02,  9.93527650833129882813e-01,  4.71063584089279174805e-01,  2.96449027197666359346e-01 ],
 [ "My gradients", "κ∇θ[3]",  8.14980268478393554688e-02,  9.55443382263183593750e-01,  5.05038917064666748047e-01,  2.77201022741208891187e-01 ],
 [    "My fields",   "u[1]",  4.31410233294131639781e-02,  9.97140933049696531754e-01,  4.62139750850942054861e-01,  3.23076684924287371725e-01 ],
 [    "My fields",   "u[2]",  1.01416659908237782872e-02,  9.14712023896926407218e-01,  4.76160523012988778913e-01,  2.71443440757963339038e-01 ],
 [    "My fields",   "θ[1]",  6.58965491052394547467e-02,  9.73216404386510802738e-01,  4.60007166313864512830e-01,  2.87310472114545079059e-01 ],
]
parr = [
 [ "My gradients", "ν∇u[1]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[2]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[3]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[4]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[5]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[6]",    16,     7,    16,     0 ],
 [ "My gradients", "κ∇θ[1]",    16,    16,    16,     0 ],
 [ "My gradients", "κ∇θ[2]",    16,    16,    16,     0 ],
 [ "My gradients", "κ∇θ[3]",    16,    16,    16,     0 ],
 [    "My fields",   "u[1]",    16,    16,    16,     0 ],
 [    "My fields",   "u[2]",    16,    16,    16,     0 ],
 [    "My fields",   "θ[1]",    16,    16,    16,     0 ],
]
#! format: on

# **Step 3.** Finally a call-back stored value can be compared for consistency to with _parr[]_ decimal places

ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))
nothing # hide

# In this trivial case the match is guaranteed. The function will return _true_ to the calling
# routine and this can be passed to an `@test` block.
#
# However we can modify the reference test values to
# see the effect of a mismatch e.g.
varr[1][3] = varr[1][3] * 10.0
ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))
nothing # hide

# Here the mis-matching field is highlighted with _N(0)_ indicating that the precision
# was not met and actual match length was (in this case) 0. If any field fails the test returns false
# for use in any regression testing control logic.
