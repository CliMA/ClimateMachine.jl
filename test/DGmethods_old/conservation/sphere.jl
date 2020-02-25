#=
Here we solve the equation:
```math
 q + dot(∇, uq) = 0
 p - dot(∇, up) = 0
```
on a sphere to test the conservation of the numerics

The boundary conditions are `p = q` when `dot(n, u) > 0` and
`q = p` when `dot(n, u) < 0` (i.e., `p` flows into `q` and vice-sersa).
=#

using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using Random

const _nstate = 2
const _q, _p = 1:_nstate
const stateid = (qid = _q, pid = _p)
const statenames = ("q", "p")

const _nauxstate = 3
@inline function velocity_initialization!(vel, x, y, z)
  @inbounds begin
    r = x^2 + y^2 + z^2
    vel[1] = cos(10*π*x) * sin(10*π*y) + cos(20 * π * z)
    vel[2] = exp(sin(π*r))
    vel[3] = sin(π * (x + y + z))
  end
end

# physical flux function
@inline function flux!(F, Q, _, vel, _)
  @inbounds begin
    u, v, w = vel[1], vel[2], vel[3]
    q, p = Q[_q], Q[_p]

    F[1, _q], F[2, _q], F[3, _q] =  u*q,  v*q,  w*q
    F[1, _p], F[2, _p], F[3, _p] = -u*p, -v*p, -w*p
  end
end
# physical flux function
@inline function numerical_flux!(F, nM, QM, _, velM, QP, _, velP, _)
  @inbounds begin
    uM, vM, wM = velM[1], velM[2], velM[3]
    unM = nM[1] * uM + nM[2] * vM + nM[3] * wM
    uP, vP, wP = velP[1], velP[2], velP[3]
    unP = nM[1] * uP + nM[2] * vP + nM[3] * wP
    un = (unP + unM) / 2

    if un > 0
      F[_q], F[_p] = un * QM[_q], -un * QP[_p]
    else
      F[_q], F[_p] = un * QP[_q], -un * QM[_p]
    end
  end
end
@inline function numerical_boundary_flux!(F, nM, QM, _, vel, QP, _, _, _, _)
  @inbounds begin
    u, v, w = vel[1], vel[2], vel[3]
    un = nM[1] * vel[1] + nM[2] * vel[2] + nM[3] * vel[3]

    if un > 0
      F[_q], F[_p] = un * QM[_q], -un * QM[_q]
    else
      F[_q], F[_p] = un * QM[_p], -un * QM[_p]
    end
  end
end

function run(mpicomm, ArrayType, N, Nhorz, Rrange, timeend, FT, dt)

  topl = StackedCubedSphereTopology(mpicomm, Nhorz, Rrange)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = Topologies.cubedshellwarp
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = flux!,
                           numerical_flux! = numerical_flux!,
                           numerical_boundary_flux! = numerical_boundary_flux!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           velocity_initialization!,
                          )

  # This is a actual state/function that lives on the grid
  initialcondition!(Q, x, y, z, _...) = begin
    @inbounds Q[_q], Q[_p] = rand(), rand()
  end
  Q = MPIStateArray(spacedisc, initialcondition!)

  lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  sum0 = weightedsum(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e
  sum(Q₀) = %.16e""" eng0 sum0

  max_mass_loss = FT(0)
  max_mass_gain = FT(0)
  cbmass = GenericCallbacks.EveryXSimulationSteps(1) do
    cbsum = weightedsum(Q)
    max_mass_loss = max(max_mass_loss, sum0 - cbsum)
    max_mass_gain = max(max_mass_gain, cbsum - sum0)
    nothing
  end
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbmass,))

  # Print some end of the simulation information
  engf = norm(Q)
  sumf = weightedsum(Q)
  @info @sprintf """Finished
  norm(Q)            = %.16e
  norm(Q) / norm(Q₀) = %.16e
  norm(Q) - norm(Q₀) = %.16e
  max mass loss      = %.16e
  max mass gain      = %.16e
  initial mass       = %.16e
  """ engf engf/eng0 engf-eng0 max_mass_loss max_mass_gain sum0
  max(max_mass_loss, max_mass_gain) / sum0
end

using Test
let
  CLIMA.init()
  ArrayTypes = (CLIMA.array_type(),)

  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))

  dt = 1e-4
  timeend = 100*dt

  polynomialorder = 4

  Nhorz = 4
  Rrange = 1.0:0.25:2.0

  dim = 3
  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64,) #Float32)
      Random.seed!(0)
      @info (ArrayType, FT, dim)
      delta_mass = run(mpicomm, ArrayType, polynomialorder, Nhorz, Rrange, timeend, FT, dt)
      @test abs(delta_mass) < 1e-15
    end
  end
end

nothing
