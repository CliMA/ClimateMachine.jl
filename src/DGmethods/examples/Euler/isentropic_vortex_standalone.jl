# Standard isentropic vortex test case.  For a more complete description of
# the setup see for Example 3 of:
#
# @article{ZHOU2003159,
#   author = {Y.C. Zhou and G.W. Wei},
#   title = {High resolution conjugate filters for the simulation of flows},
#   journal = {Journal of Computational Physics},
#   volume = {189},
#   number = {1},
#   pages = {159--179},
#   year = {2003},
#   doi = {10.1016/S0021-9991(03)00206-7},
#   url = {https://doi.org/10.1016/S0021-9991(03)00206-7},
# }
#
# This version runs the isentropic vortex as a stand alone test (no dependence
# on CLIMA moist thermodynamics)

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.MPIStateArrays

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")

# physical flux function
function eulerflux_standalone!(F, Q, ignored...)
  γ::eltype(Q) = 7 // 5
  ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  P = (γ-1)*(E - (U^2 + V^2 + W^2)/(2*ρ))

  F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
  F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
  F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
  F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
  F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
end

# initial condition
const halfperiod = 5
function isentropicvortex_standalone!(Q, t, x, y, z)
  DFloat = eltype(Q)

  γ::DFloat    = 7 // 5
  uinf::DFloat = 2
  vinf::DFloat = 1
  Tinf::DFloat = 1
  λ::DFloat    = 5

  xs = x - uinf*t
  ys = y - vinf*t

  # make the function periodic
  xtn = floor((xs+halfperiod)/(2halfperiod))
  ytn = floor((ys+halfperiod)/(2halfperiod))
  xp = xs - xtn*2*halfperiod
  yp = ys - ytn*2*halfperiod

  rsq = xp^2 + yp^2

  u = uinf - λ*(1//2)*exp(1-rsq)*yp/π
  v = vinf + λ*(1//2)*exp(1-rsq)*xp/π
  w = zero(DFloat)

  ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))
  p = ρ^γ
  U = ρ*u
  V = ρ*v
  W = ρ*w
  E = p/(γ-1) + (1//2)*ρ*(u^2 + v^2 + w^2)

  Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end

function main(mpicomm, DFloat, topl, N, endtime, ArrayType)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           nstate = _nstate,
                           flux! = eulerflux_standalone!,
                           numericalflux! = (x...) -> error())

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = isentropicvortex_standalone!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  DGBalanceLawDiscretizations.writevtk("isentropic_vortex_ic", Q, spacedisc,
                                       statenames)

end

let

  dim = 2
  DFloat = Float64
  Ne = (10, 10, 10)
  N = 4
  endtime = 10
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(-halfperiod); length=Ne[j]+1,
                               stop=halfperiod), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))
  main(mpicomm, DFloat, topl, N, endtime, ArrayType)
end

isinteractive() || MPI.Finalize()

nothing
