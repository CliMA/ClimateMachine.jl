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
# This version runs the isentropic vortex within the entire moist thermodynamics
# framework

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using Printf
using LinearAlgebra
using StaticArrays

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")


using CLIMA.PlanetParameters: cp_d, cv_d
using CLIMA.MoistThermodynamics: saturation_adjustment, air_pressure,
                                 phase_partitioning_eq, internal_energy,
                                 soundspeed_air

# preflux computation
@inline function preflux(Q, _...)
  DFloat = eltype(Q)
  ϕ::DFloat = 0
  q_t::DFloat = 0

  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W

  e_int = ρinv * E + internal_energy(0) - (u^2 + v^2 + w^2) / 2 - ϕ
  T = saturation_adjustment(e_int, ρ, q_t)
  P = air_pressure(T, ρ, q_t)

  (P, u, v, w, T)
end

# max eigenvalue
@inline function wavespeed(n, Q, G, ϕ_c, ϕ_d, t, P, u, v, w, T)
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + Q[_ρ] * soundspeed_air(T)
end

# physical flux function
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t) =
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, preflux(Q)...)

@inline function eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, P, u, v, w, T)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
  end
end

# initial condition
const halfperiod = 5
function isentropicvortex!(Q, t, x, y, z)
  DFloat = eltype(Q)

  γ::DFloat    = cp_d / cv_d
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

  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
end

function main(mpicomm, DFloat, topl::AbstractTopology{dim}, N, timeend,
              ArrayType, dt) where {dim}

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = eulerflux!,
                           numericalflux! = (x...) ->
                           NumericalFluxes.rosanuv!(x..., eulerflux!,
                                                    wavespeed,
                                                    preflux))

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = isentropicvortex!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  io = MPI.Comm_rank(mpicomm) == 0 ? stdout : devnull
  eng0 = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ (initial) =  %.16e\n", eng0)

  # Set up the information callback
  timer = [time_ns()]
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      timer[1] = time_ns()
    else
      run_time = (time_ns() - timer[1]) * 1e-9
      (min, sec) = fldmod(run_time, 60)
      (hrs, min) = fldmod(min, 60)
      @printf(io, "----\n")
      @printf(io, "simtime =  %.16e\n", ODESolvers.gettime(lsrk))
      @printf(io, "runtime =  %03d:%02d:%05.2f (hour:min:sec)\n", hrs, min, sec)
      @printf(io, "||Q||₂  =  %.16e\n", norm(Q))
    end
    nothing
  end

  #= Paraview calculators:
  P = (0.4) * (E  - (U^2 + V^2 + W^2) / (2*ρ) - 9.81 * ρ * coordsZ)
  theta = (100000/287.0024093890231) * (P / 100000)^(1/1.4) / ρ
  =#
  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
    outprefix = @sprintf("vtk/isentropicvortex_%dD_mpirank%04d_step%04d",
                         dim, MPI.Comm_rank(mpicomm), step[1])
    @printf(io, "----\n")
    @printf(io, "doing VTK output =  %s\n", outprefix)
    DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames)
    step[1] += 1
    nothing
  end

  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  Qe = MPIStateArray(spacedisc,
                    (Q, x...) -> isentropicvortex!(Q, DFloat(timeend), x...))

  # Print some end of the simulation information
  engf = norm(Q)
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ ( final ) = %.16e\n", engf)
  @printf(io, "||Q||₂ (initial) / ||Q||₂ ( final ) = %+.16e\n", engf / eng0)
  @printf(io, "||Q||₂ ( final ) - ||Q||₂ (initial) = %+.16e\n", eng0 - engf)
  @printf(io, "||Q - Qe||₂ = %.16e\n", errf)
  @printf(io, "||Q - Qe||₂ / ||Qe||₂ = %.16e\n", errf / engfe)
  errf
end

function run(dim, Ne, N, timeend, DFloat)
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(-halfperiod); length=Ne[j]+1,
                               stop=halfperiod), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))
  dt = 1e-2 / Ne[1] # not a general purpose dt calculation
  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end

using Test
let
  timeend = 0.01
  numelem = (5, 5, 1)
  lvls = 3

  polynomialorder = 4
  expected_error = Array{Float64}(undef, 2, 3) # dim-1, lvl

  # TODO: Can these be the same as standalone case?
  expected_error[1,1] = 2.7096849496802883e-02
  expected_error[1,2] = 7.3660563764448147e-03
  expected_error[1,3] = 4.2111431545573515e-04
  expected_error[2,1] = 8.5687761824661729e-02
  expected_error[2,2] = 2.3293515522772215e-02
  expected_error[2,3] = 1.3316803921428360e-03

  for DFloat in (Float64,) #Float32)
    for dim = 2:3
      err = zeros(DFloat, lvls)
      for l = 1:lvls
        err[l] = run(dim, ntuple(j->2^(l-1) * numelem[j], dim),
                     polynomialorder, timeend, DFloat)
        @test err[l] ≈ DFloat(expected_error[dim-1, l])
      end
      if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        @printf("----\n")
        for l = 1:lvls-1
          rate = log2(err[l]) - log2(err[l+1])
          @printf("rate for level %d = %e\n", l, rate)
        end
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
