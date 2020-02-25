# # Example 002: Solid Body Rotation
#
#md # !!! jupyter
#md #     This example is also available as a Jupyter notebook:
#md #     [`ex_002_solid_body_rotation.ipynb`](@__NBVIEWER_ROOT_URL__examples/DGmethods_old/generated/ex_002_solid_body_rotation.html)
#
# Key ideas of this tutorial:
#   - Setting up auxiliary state variables
#   - Defining the boundary condition treatment
#
# ## Introduction
#
# In this example we will solve the variable coefficient advection equation. The
# velocity field used is solid body rotation where the domain is the square or
# domain $\Omega = [-1, 1]^{d}$ where $d=2$ or $3$.
#
# The partial differential equation we wish to solve is
# ```math
# \frac{\partial q}{\partial t} + \nabla \cdot (\vec{u} q) = 0,
# ```
# where $q$ is the advected field and the velocity field is $\vec{u} = 2\pi r
# (-\sin(\theta), \cos(\theta), 0)^{T}$ with $r = \sqrt{x^2 + y^2}$ and $\theta
# = \arctan(y / x)$.
#
# The quantity $\vec{u} q$ is more generally called the flux and denoted in the
# tutorial below as $\boldsymbol{F}(q) = \vec{u} q$.
#
# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref ex_002_solid_body_rotation-plain-program).
#
# ## Commented Program

#------------------------------------------------------------------------------

# ### Preliminaries
# Load in modules needed for solving the problem
using MPI
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.VTK
using LinearAlgebra
using Logging
using Dates
using Printf
using StaticArrays
# Start up MPI if this has not already been done
MPI.Initialized() || MPI.Init()
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initializing the Velocity Field
# The key difference between this example and [example
# 001](ex_001_periodic_advection.html) is that in this case we have a
# non-constant velocity field. In the balance law solver, in addition to the
# PDE state at every degree of freedom we can also define a constant in time
# *auxiliary state,* and it is in this auxiliary state that we will store the
# velocity field.
#
# Initialization of the auxiliary state can happen in several ways, but here we
# will use the default `DGBalanceLaw` initialization interface which requires
# the user-defined function which given `x`, `y`, and `z` defines the auxiliary
# state
const num_aux_states = 3
function velocity_initilization!(uvec::MVector{num_aux_states, FT},
                                 x, y, z) where FT
  @inbounds begin
    r = hypot(x, y)
    θ = atan(y, x)
    uvec .= 2FT(π) * r .* (-sin(θ), cos(θ), 0)
  end
end
#md nothing # hide
# Note: We have caught the type of the elements in order the properly cast $\pi$
# since `2pi` would be default be a `Float64`.
#
# Remark: Though not needed for this problem, if the user wishes to have access
# to the coordinate points during the simulation these should be stored in the
# auxiliary state which would increase the size of the auxiliary state.

#------------------------------------------------------------------------------

# ### Physical Flux
#md # Now we define a function which given a value for $q$ computes the physical
#md # flux $\boldsymbol{F} = \vec{u} q$. The balance law solver will will pass
#md # user-defined auxiliary state at a degree of freedom through to the flux
#md # function as the fourth argument; the third and fifth arguments which are
#md # not needed for this example is the viscous state and simulation time).
function advectionflux!(F, state, _, uvec, _)
  FT = eltype(state) # get the floating point type we are using
  @inbounds begin
    q = state[1]
    F[:, 1] = uvec * q
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Numerical Flux
# As in [example 001](ex_001_periodic_advection.html) we will use an upwind
# numerical flux; more discussion of this can be seen in the [numerical
# flux](ex_001_periodic_advection.html#numerical_flux-1) section of example 001.
#
# The auxiliary state for the minus and plus sides of the interface will be
# passed in through arguments 4 and 6 of the numerical flux callback. Since the
# two sides of the interface are collocated the auxiliary state on the two sides
# should be the same.
function upwindflux!(fs, nM, stateM, viscM, uvecM, stateP, viscP, uvecP, t)
  FT = eltype(fs)
  @inbounds begin
    ## determine the advection speed and direction
    un = dot(nM, uvecM)
    qM = stateM[1]
    qP = stateP[1]
    ## Determine which state is "upwind" of the minus side
    fs[1] = un ≥ 0 ? un * qM : un * qP
  end
end

#------------------------------------------------------------------------------

# ### Boundary Numerical Flux
# Since we will not assume that the domain is periodic, we also need to define a
# boundary numerical flux which will be used to define the boundary conditions.
# Generally speaking, boundary conditions for purely hyperbolic problems should
# be imposed by relating incoming characteristic variables to outgoing
# characteristics. In this case we will use characteristic outflow boundary
# conditions and zero inflow boundary conditions.
#
# The syntax of the boundary flux is almost the same as the numerical flux
# except that the boundary condition type is passed into the function, though in
# this case we can neglect the value of the boundary condition flag. In the case
# of boundary conditions the plus state is set to the minus side state; this is
# done since in the case of model coupling this could be set to some values
# derived from the neighbouring model.
#
# In the case of advection with the outflow boundary condition and zero inflow,
# the boundary numerical flux is the same as the upwind flux except with $q^{+}$
# set to zero; more complicated PDES and boundary conditions would require more
# complex constructions.
function upwindboundaryflux!(fs, nM, stateM, viscM, uvecM, stateP, viscP, uvecP,
                             bctype, t)
  FT = eltype(fs)
  @inbounds begin
    ## determine the advection speed and direction
    un = dot(nM, uvecM)
    qM = stateM[1]
    ## Determine which state is "upwind" of the minus side
    fs[1] = un ≥ 0 ? un * qM : 0
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initial Condition
# In this example we take the initial condition to be
# ```math
# q(\vec{x}, t=0) =
# \exp\left(-\left(8\left\|\vec{x}-\frac{1}{2}\vec{e}_{1}\right\|_2\right)^2\right)
# ```
# where $\vec{e}_{1} = (1, 0, 0)^{T}$.
#
# Note: The initial condition will always be called as though the dimensionality
# of the problem is 3. For the domain used below `z = 0` when the problem is
# actually two-dimensional and thus the `hypot` call before is not effected by
# `z`
#
# Note: When the balance law solver calls the initial condition function the
# auxiliary state, in this case the velocity field, will also be included since
# the number of auxiliary variables is greater than zero.
function initialcondition!(Q, x, y, z, _)
  @inbounds Q[1] = exp(-(8 * hypot(x - 1//2, y, z))^2)
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Exact Solution
# For solid body rotation the exact solution is computed by tracing back the
# rotation to the initial state.
#
# Note: `uvec` is included to match calling convention of `initialcondition!`
function exactsolution!(Q, t, x, y, z, uvec)
  @inbounds begin
    FT = eltype(Q)

    r = hypot(x, y)
    θ = atan(y, x) - 2FT(π) * t

    x, y = r * cos(θ), r * sin(θ)

    initialcondition!(Q, x, y, z, uvec)
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initialize the DG Method
# The initialization of the DG method is largely the same as the
# [intialization](ex_001_periodic_advection.html#Initial-Condition-1) discussion
# of [ex 001](ex_001_periodic_advection.html).
function setupDG(mpicomm, dim, Ne, polynomialorder, FT=Float64,
                 ArrayType=Array)

  @assert ArrayType === Array

  brickrange = (range(FT(-1); length=Ne+1, stop=1),
                range(FT(-1); length=Ne+1, stop=1),
                range(FT(-1); length=Ne+1, stop=1))

  # By default the `BrickTopology` is not periodic, so unlike ex 001, we do not
  # need to specify the periodicity
  topology = BrickTopology(mpicomm, brickrange[1:dim])

  grid = DiscontinuousSpectralElementGrid(topology; polynomialorder =
                                          polynomialorder, FloatType = FT,
                                          DeviceArray = ArrayType,)

  # Note the additional keyword arguments: `numerical_boundary_flux!`
  # which is used to pass the numerical flux function that implements the
  # boundary condition, `auxiliary_state_length` which defines the number of
  # auxiliary state fields at each degree of freedom, and
  # `auxiliary_state_initialization!` which initializes the auxiliary state.
  spatialdiscretization = DGBalanceLaw(grid = grid, length_state_vector = 1,
                                       flux! = advectionflux!,
                                       numerical_flux! = upwindflux!,
                                       numerical_boundary_flux! =
                                       upwindboundaryflux!,
                                       auxiliary_state_length = num_aux_states,
                                       auxiliary_state_initialization! =
                                       velocity_initilization!)

end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initializing and run the DG method
# This `let` statement is largely the same as the [Using ODE solver callback
# functions](ex_001_periodic_advection.html#Using-ODE-solver-callback-functions-1)
# block from ex 001. Difference are highlighted.
let
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)
  dim = 2
  Ne = 20
  polynomialorder = 4
  spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder)
  Q = MPIStateArray(spatialdiscretization, initialcondition!)
  filename = @sprintf("initialcondition_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))

  h = 1 / Ne
  # Since we are on the $[-1, 1]^d$ domain, the maximum velocity will by $2\pi$,
  # thus this defines the CFL restriction
  CFL = h / (2π)
  dt = CFL / polynomialorder^2
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)
  finaltime = 1.0
  if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
      && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
    parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
    finaltime = 2dt                                           #src
  end                                                         #src

  # For simplicity we only include the vtk callback

  vtk_step = 0
  mkpath("vtk")
  cb_vtk = GenericCallbacks.EveryXSimulationSteps(20) do
    vtk_step += 1
    filename = @sprintf("vtk/solid_body_rotation_mpirank%04d_step%04d",
                         MPI.Comm_rank(mpicomm), vtk_step)
    writevtk(filename, Q, spatialdiscretization,
                                         ("q",))
    nothing
  end

  solve!(Q, lsrk; timeend = finaltime, callbacks = (cb_vtk, ))

  filename = @sprintf("finalsolution_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))

  # As with the initial condition, we need to catch the auxiliary state `uvec`
  # in this initialization call.
  Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, uvec
    exactsolution!(Qin, finaltime, x, y, z, uvec)
  end

  error = euclidean_distance(Q, Qe)
  with_logger(mpi_logger) do
    @info @sprintf("""Run with
                   dim              = %d
                   Ne               = %d
                   polynomial order = %d
                   error            = %e
                   """, dim, Ne, polynomialorder, error)
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Computing rates and errors
# As with ex 001, since the analytic solution is known we can compute the rate
# of convergence of the scheme
let
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)

  dim = 2
  polynomialorder = 4
  finaltime = 1.0

  with_logger(mpi_logger) do
    @info @sprintf("""Running with
                   dim              = %d
                   polynomial order = %d
                   """, dim, polynomialorder)
  end

  base_Ne = 5
  lvl_error = zeros(4) # number of levels to compute is length(lvl_error)
  for lvl = 1:length(lvl_error)
    ## `Ne` for this mesh level
    Ne = base_Ne * 2^(lvl-1)
    spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder)

    Q = MPIStateArray(spatialdiscretization, initialcondition!)
    h = 1 / Ne
    CFL = h / (2π)
    dt = CFL / polynomialorder^2
    if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
        && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
      parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
      finaltime = 2dt                                           #src
    end                                                         #src
    lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

    solve!(Q, lsrk; timeend = finaltime)

    Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, uvec
      exactsolution!(Qin, finaltime, x, y, z, uvec)
    end

    lvl_error[lvl] = euclidean_distance(Q, Qe)
    msg =  @sprintf   "Level      = %d" lvl
    msg *= @sprintf "\nNe               = %d" Ne
    msg *= @sprintf "\nerror            = %.4e" lvl_error[lvl]
    if lvl > 1
      rate = log2(lvl_error[lvl-1]) - log2(lvl_error[lvl])
      msg *= @sprintf "\nconvergence rate = %.4e" rate
    end
    with_logger(mpi_logger) do
      @info msg
    end
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

#md # ## [Plain Program](@id ex_002_solid_body_rotation-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ex\_002\_solid\_body\_rotation\_periodic\_advection.jl](ex_002_solid_body_rotation.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
