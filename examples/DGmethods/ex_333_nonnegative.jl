using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Vtk
using LinearAlgebra
using Logging
using Dates
using Printf
using StaticArrays

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const DeviceArrayType = CuArray
else
  const DeviceArrayType = Array
end

MPI.Initialized() || MPI.Init()

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
  DFloat = eltype(Q)
  x0, y0 = DFloat(1//4), DFloat(1//4)
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
end

function setupDG(mpicomm, dim, Ne, polynomialorder, DFloat=Float64, ArrayType=Array)
  brickrange = (range(DFloat(0); length=Ne+1, stop=1),
                range(DFloat(0); length=Ne+1, stop=1),
                range(DFloat(0); length=Ne+1, stop=1))

  topology = BrickTopology(mpicomm, brickrange[1:dim])

  grid = DiscontinuousSpectralElementGrid(topology; polynomialorder =
                                          polynomialorder, FloatType = DFloat,
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
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)
  rank = MPI.Comm_rank(mpicomm)

  @static if haspkg("CUDAnative")
    device!(rank % length(devices()))
  end

  dim = 2
  Ne = 20
  polynomialorder = 4
  DFloat = Float64

  spatialdiscretization = setupDG(mpicomm, dim, Ne, polynomialorder, DFloat,
                                  DeviceArrayType)
  Q = MPIStateArray(spatialdiscretization, initialcondition!)

  h = 1 / Ne
  CFL = h / 1
  dt = CFL / polynomialorder^2
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  vtk_step = 0
  mkpath("vtk")
  function vtkoutput()
    filename = @sprintf("vtk/q_rank%04d_step%04d", rank, vtk_step)
    writevtk(filename, Q, spatialdiscretization, ("q",))

    minQ = MPI.Reduce([minimum(Q.realQ)], MPI.MIN, 0, Q.mpicomm)[1]
    maxQ = MPI.Reduce([maximum(Q.realQ)], MPI.MAX, 0, Q.mpicomm)[1]
    sumQ = weightedsum(Q)

    with_logger(mpi_logger) do
      @info @sprintf("""Run with
                     min Q = %25.16e
                     max Q = %25.16e
                     sum Q = %25.16e
                     """, minQ, maxQ, sumQ)
    end

    vtk_step += 1
    nothing
  end

  cb_vtk = GenericCallbacks.EveryXSimulationSteps(vtkoutput, 20)

  vtkoutput()

  # We integrate so that the final solution is equal to the initial solution
  Qe = copy(Q)

  solve!(Q, lsrk; timeend = finaltime, callbacks = (cb_vtk, ))

  vtkoutput()

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

run()
