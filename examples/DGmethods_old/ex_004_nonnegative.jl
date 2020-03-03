# This example uses the TMAR Filter from
#
#    @article{doi:10.1175/MWR-D-16-0220.1,
#      author = {Light, Devin and Durran, Dale},
#      title = {Preserving Nonnegativity in Discontinuous Galerkin
#               Approximations to Scalar Transport via Truncation and Mass
#               Aware Rescaling (TMAR)},
#      journal = {Monthly Weather Review},
#      volume = {144},
#      number = {12},
#      pages = {4771-4786},
#      year = {2016},
#      doi = {10.1175/MWR-D-16-0220.1},
#    }
#
# to reproduce the example in section 4b.  It is a shear swirling
# flow deformation of a transported quantity from LeVeque 1996.  The exact
# solution at the final time is the same as the initial condition.
#
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Mesh.Filters
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

const finaltime = 5

function velocity(x, y, t)
  sx, cx = sinpi(x), cospi(x)
  sy, cy = sinpi(y), cospi(y)
  ct = cospi(t/finaltime)

  u =  2sx^2*sy*cy*ct
  v = -2sy^2*sx*cx*ct
  (u, v)
end

cosbell(τ, q) = τ ≤ 1 ? ((1 + cospi(τ))/2)^q : zero(τ)

function initialcondition!(Q, x, y, z, _)
  FT = eltype(Q)
  x0, y0 = FT(1//4), FT(1//4)
  τ = 4hypot(x-x0, y-y0)
  @inbounds Q[1] = cosbell(τ, 3)
end

const num_aux_states = 4
const _a_x, _a_y, _a_u, _a_v = 1:num_aux_states

function aux_init!(aux, x, y, z)
  u, v = velocity(x, y, zero(x))

  @inbounds aux[_a_x], aux[_a_y], aux[_a_u], aux[_a_v] = x, y, u, v
end

function advectionflux!(F, state, _, aux, _)
  @inbounds begin
    U, q = @SVector([aux[_a_u], aux[_a_v], 0]), state[1]
    F[:, 1] .= U*q
  end
end

function upwindflux!(fs, nM, stateM, _, auxM, stateP, viscP, auxP, t)
  @inbounds begin
    UM, qM, qP = @SVector([auxM[_a_u], auxM[_a_v], 0]), stateM[1], stateP[1]
    un = dot(nM, UM)

    fs[1] = un ≥ 0 ? un * qM : un * qP
  end
end

function upwindboundaryflux!(fs, nM, stateM, viscM, auxM, stateP, viscP, auxP,
                             bctype, t)
  stateP .= 0
  upwindflux!(fs, nM, stateM, viscM, auxM, stateP, viscP, auxP, t)
end

function preodefun!(disc, Q, t)
  # Update the velocity
  DGBalanceLawDiscretizations.dof_iteration!(disc.auxstate, disc, Q) do R, _, _, aux
      @inbounds R[_a_u], R[_a_v] = velocity(aux[_a_x], aux[_a_y], t)
  end

  Filters.apply!(Q, 1, disc.grid, TMARFilter())
end

function setupDG(mpicomm, dim, Ne, polynomialorder, FT=Float64, ArrayType=Array)
  brickrange = (range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=Ne+1, stop=1),
                range(FT(0); length=Ne+1, stop=1))

  topology = BrickTopology(mpicomm, brickrange[1:dim])

  grid = DiscontinuousSpectralElementGrid(topology; polynomialorder =
                                          polynomialorder, FloatType = FT,
                                          DeviceArray = ArrayType,)

  spatialdiscretization = DGBalanceLaw(grid = grid, length_state_vector = 1,
                                       flux! = advectionflux!,
                                       numerical_flux! = upwindflux!,
                                       numerical_boundary_flux! =
                                       upwindboundaryflux!,
                                       auxiliary_state_length = num_aux_states,
                                       auxiliary_state_initialization! = aux_init!,
                                       preodefun! = preodefun!)
end

function run()
  CLIMA.init()
  DeviceArrayType = CLIMA.array_type()

  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)
  rank = MPI.Comm_rank(mpicomm)

  dim = 2
  Ne = 20
  polynomialorder = 4
  FT = Float64

  spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder, FT,
                                  DeviceArrayType)
  Q = MPIStateArray(spatialdiscretization, initialcondition!)

  maxvelosity = 2
  elementsize = 1 / Ne
  dx = elementsize / polynomialorder^2
  CFL = 1

  dt = CFL * dx / maxvelosity
  @info @sprintf "dt = %1.2e" dt
  sork = SSPRK33ShuOsher(spatialdiscretization, Q; dt = dt, t0 = 0)

  initialsumQ = weightedsum(Q)

  vtk_step = 0
  mkpath("vtk")
  function vtkoutput()
    Filters.apply!(Q, 1, spatialdiscretization.grid, TMARFilter())

    filename = @sprintf("vtk/q_rank%04d_step%04d", rank, vtk_step)
    writevtk(filename, Q, spatialdiscretization, ("q",))

    minQ = MPI.Reduce([minimum(Q.realdata)], MPI.MIN, 0, Q.mpicomm)
    maxQ = MPI.Reduce([maximum(Q.realdata)], MPI.MAX, 0, Q.mpicomm)
    sumQ = weightedsum(Q)

    with_logger(mpi_logger) do
      @info @sprintf("""step = %d
                           min Q = %25.16e
                           max Q = %25.16e
                     sum error Q = %25.16e
                     """, vtk_step, minQ[1], maxQ[1], (initialsumQ -
                                                       sumQ)/initialsumQ)
    end

    vtk_step += 1
    nothing
  end

  cb_vtk = GenericCallbacks.EveryXSimulationSteps(vtkoutput, 40)

  vtkoutput()

  # We integrate so that the final solution is equal to the initial solution
  Qe = copy(Q)

  solve!(Q, sork; timeend = finaltime, callbacks = (cb_vtk, ))

  vtkoutput()

  minQ = MPI.Reduce([minimum(Q.realdata)], MPI.MIN, 0, Q.mpicomm)
  maxQ = MPI.Reduce([maximum(Q.realdata)], MPI.MAX, 0, Q.mpicomm)
  finalsumQ = weightedsum(Q)
  sumerror = (initialsumQ - finalsumQ) / initialsumQ
  error = euclidean_distance(Q, Qe)
  with_logger(mpi_logger) do
    @info @sprintf("""Run with
                   dim              = %d
                   Ne               = %d
                   polynomial order = %d
                   min              = %e
                   max              = %e
                   L2 error         = %e
                   sum error        = %e
                   """, dim, Ne, polynomialorder, minQ[1], maxQ[1], error,
                   sumerror)
  end
end

run()
