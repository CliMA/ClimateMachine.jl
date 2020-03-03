# # Example 001: Periodic Advection
#
#md # !!! jupyter
#md #     This example is also available as a Jupyter notebook:
#md #     [`ex_001_periodic_advection.ipynb`](@__NBVIEWER_ROOT_URL__examples/DGmethods_old/generated/ex_001_periodic_advection.html)
#
# Key ideas of this tutorial:
#   - Setting up PDE
#   - Defining a numerical flux
#   - Defining finite element mesh
#   - Using the ODE solving framework
#   - Using VTK visualization
#
# ## Introduction
#
# In this example we will solve the constant coefficient advection equation on a
# periodic domain; the domain is taken to be the unit square or cube depending on
# whether the problem is two- or three-dimensional.
#
# The partial differential equation we wish to solve is
#
# ```math
#  \frac{\partial q}{\partial t} + \nabla \cdot (\vec{u} q) = 0,
# ```
#
# where $q$ is the advected scalar quantity and $\vec{u}$ is the constant
# velocity field. The quantity $\vec{u} q$ is more generally called the flux and
# denoted in the tutorial below as $\boldsymbol{F}(q) = \vec{u} q$.
#
# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref ex_001_periodic_advection-plain-program).
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

# define the velocity field for advection
const uvec = (1, 2, 3)
#md nothing # hide

#------------------------------------------------------------------------------

# ### Physical Flux
#md # Now we define a function which given a value for $q$ computes the physical
#md # flux $\boldsymbol{F} = \vec{u} q$. Since we only have a single state, $q$,
#md # the state will come in as an `MVector` of length `1` and the flux to fill
#md # as an `MMatrix` of size `(3, 1)`.
#
#md # Note: for two-dimensional simulations the flux `MMatrix` will also be of
#md # size `(3, 1)` but the function need only fill the first two rows
function advectionflux!(F, state, _...)
  FT = eltype(state) # get the floating point type we are using
  @inbounds begin
    q = state[1]
    F[:, 1] = SVector{3, FT}(uvec) * q
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Numerical Flux
# In the discontinuous Galerkin method the continuity of the solution across
# element interfaces is imposed weakly through the use of a numerical flux. The
# numerical flux is a function that given the solution state on either side of
# an interface returns a unique value approximating $\boldsymbol{F}\cdot
# \vec{n}$ on the interface:
# ```math
# f^{*} = f^{*}(q^{-}, q^{+}; \vec{n}).
# ```
# We call the two sides of the interface the "minus side" and "plus side" hence
# the $\pm$ superscripts in $q$ in the notation above; the choice of which
# element is on the minus side and plus side is arbitrary. Here $\vec{n}$ is a
# unit normal to the interface and is typically taken to point from the minus
# side to the plus side. The numerical flux is required to be symmetric with
# respect to $q^{-}$ and $q^{+}$:
# ```math
# f^{*}(q^{-}, q^{+}; \vec{n}) = f^{*}(q^{+}, q^{-}; \vec{n}).
# ```
# and consistent with the physical flux in the sense that
# ```math
# \boldsymbol{F}(q) \cdot \vec{n} = f^{*}(q, q; \vec{n}).
# ```
# Taken together these two conditions also imply that
# ```math
# f^{*}(q^{-}, q^{+}; \vec{n}^{-}) = -f^{*}(q^{+}, q^{-}; \vec{n}^{+}),
# ```
# that is $f^{*}$ is skew-symmetry with respect to the unit normals $\vec{n}^{+}
# = -\vec{n}^{-}$.  The choice of numerical flux has important implications for
# the stability of the method. Though it is beyond the scope of these tutorials
# to dive into this in detail, often the "best" numerical fluxes are the ones
# that are constructed specifically for given set of equations by solving the
# Riemann problem; the Riemann problem is an initial value problem where the
# initial data are piecewise constant with a single discontinuity.
#
# In the CLIMA balance law solver the numerical flux function is a user-defined
# function that fills in an `MVector` for the numerical flux given two states,
# the "viscous state", a unit normal to the face, the simulation time, and a
# user-defined auxiliary state; the viscous and auxiliary states will be
# discussed in a subsequent examples. In the function below `F` is the numerical
# flux to fill, `nM` is the unit normal pointing away from the minus side and
# toward the plus side, `QM` and `QP` are `MVector`s of the solution state on
# the minus and plus sides of the interface, `viscM` and `viscP` are the viscous
# states on the minus and plus sides, `auxM` and `auxP` are the user-defined
# auxiliary state values on the minus and plus sides, and `t` is the simulation
# time.
#
# For linear advection the solution to the Riemann problem is trivial, since
# if $\vec{n}^{-} \cdot \vec{u} ≥ 0$ the state $q$ is being advected from
# the minus side to the plus side otherwise the reverse occurs. Thus the upwind
# numerical flux for advection is
# ```math
# f^{*}(q^{-}, q^{+}; \vec{n}^{-}) =
# \begin{cases}
#   \vec{n}^{-}\cdot\vec{u} \; q^{-}, &\text{ if } \vec{n} \cdot \vec{u} ≥ 0,\\
#   \vec{n}^{-}\cdot\vec{u} \; q^{+}, &\text{ otherwise.}
# \end{cases}
# ```
# This is done in the following function
function upwindflux!(fs, nM, stateM, viscM, auxM, stateP, viscP, auxP, t)
  FT = eltype(fs)
  @inbounds begin
    ## determine the advection speed and direction
    un = dot(nM, FT.(uvec))
    qM = stateM[1]
    qP = stateP[1]
    ## Determine which state is "upwind" of the minus side
    fs[1] = un ≥ 0 ? un * qM : un * qP
  end
end
#md nothing # hide

# In later examples we will demonstrate how to use the Rusanov flux which is
# included in the `CLIMA.DGBalanceLawDiscretizations.NumericalFluxes` submodule.
# This is a more general-purpose flux which approximates the solution Riemann
# problem by using an average of the flux on either side of the interface with
# additional dissipation added based on the local wave speed.

#------------------------------------------------------------------------------

# ### Initial Condition
# In this example we take the initial condition to be
# ```math
# q(\vec{x}, t=0) = \prod_{i=1}^{d} \exp(\sin(2\pi x_{i})),
# ```
# where $d$ is the dimensionality of the problem. To use this initial condition
# we need a function that given $\vec{x}$ sets $q$.
#
# The initial condition is set by the solver through a function which takes $q$
# as an `MVector` to initialize based on the pointwise coordinate location
# `(x_1, x_2, x_3)`.
#
# Note: The initial condition will always be called as though the dimensionality
# of the problem is 3. For the domain used below `x_3 = 0` when the problem is
# actually two-dimensional; since when $x_3 = 0$ the function $\exp(\sin(2\pi
# x_{3})) = 1$ we can safely assume the dimensionality is always $3$ in our
# implication of the initial condition.
#
# Note: The last argument needs to be caught but not used for this example
function initialcondition!(Q, x_1, x_2, x_3, _...)
  @inbounds Q[1] = exp(sin(2π * x_1)) * exp(sin(2π * x_2)) * exp(sin(2π * x_3))
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Exact Solution
# For periodic constant-velocity advection the exact solution is trivial to
# compute. Assuming that $\phi(x)$ is the periodically replicated initial
# condition, the analytic solution is
# ```math
# q(\vec{x}, t) = \phi(\vec{x} - \vec{u} t).
# ```
# This will be useful later since it will allow us to check our work by
# computing the error in our solution and estimating the convergence rate.
#
# For a general initial condition on the unit domain the following function can
# be used:
function exactsolution!(dim, Q, t, x_1, x_2, x_3, _...)
  @inbounds begin
    FT = eltype(Q)

    ## trace back the point (x_1, x_2, x_3) in the velocity field and
    ## determine where in our "original" [0, L_1] X [0, L_2] X [0, L_3] domain
    ## this point is located
    y_1 = mod(x_1 - FT(uvec[1]) * t, 1)
    y_2 = mod(x_2 - FT(uvec[2]) * t, 1)

    ## if we are really just 2-D we do not want to change the x_3 coordinate
    y_3 = dim == 3 ? mod(x_3 - FT(uvec[3]) * t, 1) : x_3

    initialcondition!(Q, y_1, y_2, y_3)
  end
end
#md nothing # hide
# The input argument `dim` is the "real" dimensionality of the problem and is
# needed in case `uvec[3] != 0`.

#------------------------------------------------------------------------------

# ### Initialize the DG Method
# We are now at the point that we can initialize the structure for the DG
# method.  For convenience we define a function that initializes the DG method
# over a given MPI communicator `mpicomm`, for a given `polynomialorder`, using
# `dim` dimensions. The mesh used will be `Ne` x `Ne` elements when `dim == 2`
# and `Ne` x `Ne` x `Ne` elements when `dim == 3`. The floating point type of
# the computation is `FT` and whether the CPU or GPU is used is determined
# by `ArrayType`; `ArrayType === Array` is for the CPU and `ArrayType ===
# CuArray` is for NVIDIA GPUs.
#
# Note: This whole code chunk is in a function block
function setupDG(mpicomm, dim, Ne, polynomialorder, FT=Float64,
                 ArrayType=Array)

  @assert ArrayType === Array

  # We will use the `BrickTopology` from `CLIMA.Mesh.Topologies` to define the mesh.
  # The "topology" in CLIMA is the element connectivity information (e.g.,
  # neighbouring elements and interface data) along with coordinate locations
  # for corners of the elements. The `BrickTopology` creates a regular mesh of a
  # rectangular or regular hexahedral domain. This is done by specifying the
  # coordinate points of the element corners along each dimension. Here, we want
  # to mesh the unit square or cube with `Ne` elements in each dimension, thus
  # we specify the following `Tuple` of `range`s:
  brickrange = (range(FT(0); length=Ne+1, stop=1), # x_1 corner locations
                range(FT(0); length=Ne+1, stop=1), # x_2 corner locations
                range(FT(0); length=Ne+1, stop=1)) # x_3 corner locations
  # these coordinates will be combined in a tensor product fashion to define the
  # element corners.
  #
  # By default the `BrickTopology` is not periodic, so we need to define a
  # `Tuple` of boolean defining which dimensions are periodic.
  periodicity = (true, true, true)
  # Note: We have defined both the `brickrange` and `periodicity` as though we
  # are working in three-dimensions, and in when in two-dimensions we will
  # discard the third element of the `Tuple`; this could also be done from the
  # start by using the `ntuple` function.

  # Using `brickrange` and `periodicity` we can now initialize the topology
  # using the `BrickTopology` constructor. This will both create the topology as
  # well as do the partitioning of the elements across the MPI ranks available
  # in the mpi communicator `mpicomm`
  topology = BrickTopology(mpicomm, brickrange[1:dim];
                           periodicity=periodicity[1:dim])

  # The topology only has element connectivity and corner information, thus we
  # still need to create a grid (or mesh) of degrees of freedom. In CLIMA the
  # so-called discontinuous spectral element grid is used (aka tensor-product
  # quadrilateral and hexahedral elements with Legendre-Gauss-Lobatto
  # interpolation and quadrature weights). Given a topology and polynomial order
  # we can create the grid of degrees of free using
  grid = DiscontinuousSpectralElementGrid(topology; polynomialorder =
                                          polynomialorder, FloatType = FT,
                                          DeviceArray = ArrayType,)
  # Note: This constructor also takes in a `FloatType` which specifies the
  # floating point type to be used (e.g., for the coordinate points and geometry
  # metric terms). The argument `ArrayType` is used to determine the compute
  # device to use (i.e., `Array` will signify the CPU is being used and
  # `CuArray` will signal that an NVIDIA GPU is being used).

  # We can now define the discretization from the `grid` using the physical flux
  # `advectionflux!` and numerical flux `upwindflux!` defined above; we only
  # have a single state variable $q$ hence `length_state_vector = 1`
  spatialdiscretization = DGBalanceLaw(grid = grid, length_state_vector = 1,
                                       flux! = advectionflux!,
                                       numerical_flux! = upwindflux!)

  # (end of function)
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initializing and run the DG method
# Note: This whole code chunk is in a `let` block
let
  # We will just use the whole MPI communicator
  mpicomm = MPI.COMM_WORLD

  # Since this is an MPI enabled code, we use the Julia logging functionality to
  # ensure that only one MPI rank prints to the screen. Namely, MPI rank 0 does
  # all the logging and all other ranks dump their ouput to `devnull`.
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)
  # Note: The `NullLogger` should not be used because if any MPI functions are
  # called in a logging block deadlock will occur since `NullLogger` code is
  # not executed.

  # Dimensionality to run
  dim = 2

  # Mesh size along each dimension
  Ne = 20

  # order of polynomials to use
  polynomialorder = 4

  # Setup the DG discretization
  spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder)

  # Given the `spatialdiscretization` and the `initialcondition!` function we
  # can create and initialize storage for the solution. This is an MPI-aware
  # array
  Q = MPIStateArray(spatialdiscretization, initialcondition!)

  # A VTK file, which can be viewed in [ParaView](https://www.paraview.org/) or
  # [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit), can be
  # generated with the following command (the last `Tuple` gives strings for the
  # names of the state fields)
  filename = @sprintf("initialcondition_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))
  # Note: Currently the `writevtk` function writes one file for each MPI rank,
  # and the user is responsible for opening each of the files to "stitch"
  # together the image.

  # In order to run the simulation we need to use an ODE solver. In this example
  # we will use a low storage Runge-Kutta method which can be initialized with a
  # spatial discretization, solution vector (not stored but used to define
  # needed auxiliary arrays), initial solution time, and a time step size; this
  # particular Runge-Kutta method uses a fixed time step.
  #
  # Since we are using a regular mesh, with a constant wave speed, a "CFL"
  # restriction for the mesh is
  h = 1 / Ne                           # element size
  CFL = h / maximum(abs.(uvec[1:dim])) # time to cross the element once
  dt = CFL / polynomialorder^2         # DG time step scaling (for this
                                       ## particular RK scheme could go with a
                                       ## factor of ~2 larger time step)
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  # Here we run the ODE solver until the final time `timeend` using `Q` as the
  # initial condition `Q`. The solution will be updated in place so that the
  # final solution will also be stored in `Q`.
  finaltime = 1.0
  if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
      && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
    parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
    finaltime = 2dt                                           #src
  end                                                         #src
  solve!(Q, lsrk; timeend = finaltime)

  # The final solution can be visualized in a similar manner to the initial
  # condition
  filename = @sprintf("finalsolution_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))

  # Using the `finaltime` and `exactsolution!` we can calculate the exact
  # solution
  Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, aux
    exactsolution!(dim, Qin, finaltime, x, y, z)
  end
  # and then compute the error by evaluating the Euclidean distance between the
  # computed solution `Q` and the exact solution `Qe`
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

# ### Using ODE solver callback functions
# The above simulation run with `solve!` runs from the initial time to the final
# time. The ODE solver framework in CLIMA gives functionality that allows the
# user to *inject* code into the solver during execution. Here we show how to
# use some of the generic callback functions provided to
#
#  - Save diagnostic information
#  - display runtime simulation information
#  - save VTK files during the simulation
#
# Note: This whole code chunk is in a `let` block
let
  # code is the same as above until the `solve!` call
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
  CFL = h / maximum(abs.(uvec[1:dim]))
  dt = CFL / polynomialorder^2
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)
  finaltime = 1.0
  if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
      && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
    parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
    finaltime = 2dt                                           #src
  end                                                         #src

  # The ODE solver callback functions are called both before the ODE solver
  # begins and then after each time step.
  #
  # For instance if a user wanted to store the norm of the solution every time
  # step the following callback could be used
  store_norm_index = 0
  normQ = Array{Float64}(undef, ceil(Int, finaltime / dt))
  function cb_store_norm()
    store_norm_index += 1
    normQ[store_norm_index] = norm(Q)
    nothing
  end
  # Note: that callbacks must return either `nothing` or `0` if the ODE solver
  # should continue, `1` is the ODE solver should stop after all the callbacks
  # have executed, or `2` is the time stepping should immediately stop with no
  # further callbacks executed.

  # Several generic callbacks are provided in the `CLIMA.GenericCallbacks`
  # submodule. For instance, the `EveryXSimulationSteps` callbacks will execute
  # every `X` time steps. This could be used to say write VTK output every `20`
  # time steps
  vtk_step = 0
  mkpath("vtk")
  cb_vtk = GenericCallbacks.EveryXSimulationSteps(20) do
    vtk_step += 1
    filename = @sprintf("vtk/advection_mpirank%04d_step%04d",
                         MPI.Comm_rank(mpicomm), vtk_step)
    writevtk(filename, Q, spatialdiscretization,
                                         ("q",))
    nothing
  end

  # Another provided generic callback is `EveryXWallTimeSeconds` which will be
  # called every `X` seconds of wall clock time (as opposed to simulation time).
  # This could be used to dump diagnostic information about the simulation. In
  # this case we display the norm of the simulation time, the run time, and the
  # norm of the solution.
  #
  # One unique feature of this call back is that it takes in a single optional
  # argument `init` which allows the ODE solver to call the callback for
  # initialization; all callbacks get called for initialization with a single
  # boolean argument set to `true`, but this occurs in a `try/catch` statement
  # in case the callback does not require initialization (such as the two
  # above).
  starttime = Ref(now())
  cb_info = GenericCallbacks.EveryXWallTimeSeconds(1, mpicomm) do (init=false)
    if init
      starttime[] = now()
    else
      with_logger(mpi_logger) do
        @info @sprintf("""Update
                       simtime = %.16e
                       runtime = %s
                       norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                       Dates.format(convert(Dates.DateTime,
                                            Dates.now()-starttime[]),
                                    Dates.dateformat"HH:MM:SS"),
                       norm(Q))
      end
    end
  end
  # Note that this callback also takes in the MPI communicator. This is
  # necessary because the callback needs to execute an `MPI.Allreduce` to ensure
  # that all the MPI ranks are using the same global run time.

  # the defined callbacks are based to the ODE `solve!` function through the
  # keyword argument `callbacks` as a tuple:
  solve!(Q, lsrk; timeend = finaltime,
         callbacks = (cb_store_norm, cb_vtk, cb_info))

  # The remainder of the function is the same as above
  filename = @sprintf("finalsolution_mpirank%04d", MPI.Comm_rank(mpicomm))
  writevtk(filename, Q, spatialdiscretization,
                                       ("q",))

  Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, aux
    exactsolution!(dim, Qin, finaltime, x, y, z)
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
# If the above code is put in a loop over increasing `Ne` then a rate of
# convergence for the scheme can be established which we expect to be
# on the order of the polynomial order plus $\sim 1/2$.
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
    CFL = h / maximum(abs.(uvec[1:dim]))
    dt = CFL / polynomialorder^2
    if (parse(Bool, lowercase(get(ENV,"TRAVIS","false")))       #src
        && "Test" == get(ENV,"TRAVIS_BUILD_STAGE_NAME","")) ||  #src
      parse(Bool, lowercase(get(ENV,"APPVEYOR","false")))       #src
      finaltime = 2dt                                           #src
    end                                                         #src
    lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

    solve!(Q, lsrk; timeend = finaltime)

    Qe = MPIStateArray(spatialdiscretization) do Qin, x, y, z, aux
      exactsolution!(dim, Qin, finaltime, x, y, z)
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

#md # ## [Plain Program](@id ex_001_periodic_advection-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ex\_001\_periodic\_advection.jl](ex_001_periodic_advection.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
