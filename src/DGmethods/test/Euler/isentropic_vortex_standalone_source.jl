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
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")
const γ_exact = 7 // 5
if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  using Random
end

# preflux computation
@inline function preflux(Q, _...)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2), u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t, P, u, v, w, ρinv)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

const _nauxstate = 7
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_x, _a_y, _a_z = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    aux[_a_ϕ] = hypot(x, y, z)
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
  end
end

@inline function almost_no_source!(S, Q, aux, t)
  @inbounds begin
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    isentropicvortex!(S, t, x, y, z)
    ρ, ρe = Q[_ρ], S[_ρ]
    S[_ρ] = 0
    S[_U] = -abs(aux[_a_ϕx]) * (ρe - ρ)
    S[_V] = -abs(aux[_a_ϕy]) * (ρe - ρ)
    S[_W] = -abs(aux[_a_ϕz]) * (ρe - ρ)
    S[_E] = 0
  end
end

# physical flux function
eulerflux!(F, Q, QV, aux, t) =
eulerflux!(F, Q, QV, aux, t, preflux(Q)...)

@inline function eulerflux!(F, Q, QV, aux, t, P, u, v, w, ρinv)
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
function isentropicvortex!(Q, t, x, y, z, _...)
  DFloat = eltype(Q)

  γ::DFloat    = γ_exact
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

  if integration_testing
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
  else
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] =
    10+rand(), rand(), rand(), rand(), 10+rand()
  end

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
                           numerical_flux! = (x...) ->
                           NumericalFluxes.rusanov!(x..., eulerflux!,
                                                    wavespeed,
                                                    preflux),
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           auxiliary_state_initialization!,
                           source! = almost_no_source!)

  DGBalanceLawDiscretizations.grad_auxiliary_state!(spacedisc, _a_ϕ,
                                                    (_a_ϕx, _a_ϕy, _a_ϕz))

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = isentropicvortex!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf """Update
  simtime = %.16e
  runtime = %s
  norm(Q) = %.16e""" ODESolvers.gettime(lsrk) Dates.format(convert(Dates.DateTime, Dates.now()-starttime[]), Dates.dateformat"HH:MM:SS") energy
    end
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
    @debug "doing VTK output" outprefix
    DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames)
    step[1] += 1
    nothing
  end

  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  if integration_testing
    Qe = MPIStateArray(spacedisc,
                       (Q, x...) -> isentropicvortex!(Q, DFloat(timeend), x...))
    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf/eng0 engf-eng0 errf errf / engfe
  else
    @info @sprintf """Finished
    norm(Q)            = %.16e
    norm(Q) / norm(Q₀) = %.16e
    norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0
  end
  integration_testing ? errf : (engf / eng0)
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)
  ArrayType = Array

  brickrange = ntuple(j->range(DFloat(-halfperiod); length=Ne[j]+1,
                               stop=halfperiod), dim)
  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(j->true, dim))
  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
end

using Test
let
  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
  mpicomm = MPI.COMM_WORLD
  if MPI.Comm_rank(mpicomm) == 0
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
    global_logger(ConsoleLogger(stderr, loglevel))
  else
    global_logger(NullLogger())
  end

  if integration_testing
    timeend = 1
    numelem = (5, 5, 1)

    polynomialorder = 4

    expected_error = Array{Float64}(undef, 2, 3) # dim-1, lvl
    expected_error[1,1] = 5.5175136745319797e-01
    expected_error[1,2] = 6.7757089928958958e-02
    expected_error[1,3] = 3.2832015676432292e-03
    expected_error[2,1] = 1.7613313745738965e+00
    expected_error[2,2] = 2.1526080821361515e-01
    expected_error[2,3] = 1.0251374591125394e-02
    lvls = size(expected_error, 2)

    for DFloat in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(DFloat, lvls)
        for l = 1:lvls
          Ne = ntuple(j->2^(l-1) * numelem[j], dim)
          dt = 1e-2 / Ne[1]
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps
          err[l] = run(mpicomm, dim, Ne, polynomialorder, timeend, DFloat, dt)
          @test err[l] ≈ DFloat(expected_error[dim-1, l])
        end
        @info begin
          msg = ""
          for l = 1:lvls-1
            rate = log2(err[l]) - log2(err[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  else
    numelem = (3, 4, 5)
    dt = 1e-3
    timeend = 2dt

    polynomialorder = 4

    mpicomm = MPI.COMM_WORLD

    check_engf_eng0 = Dict{Tuple{Int64, Int64, DataType}, AbstractFloat}()
    check_engf_eng0[2, 1, Float64] = 9.9999808508887378e-01
    check_engf_eng0[3, 1, Float64] = 9.9999644038110480e-01
    check_engf_eng0[2, 3, Float64] = 9.9999878540546705e-01
    check_engf_eng0[3, 3, Float64] = 9.9999657187253710e-01

    for DFloat in (Float64,) #Float32)
      for dim = 2:3
        Random.seed!(0)
        engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                        DFloat, dt)
        @test check_engf_eng0[dim, MPI.Comm_size(mpicomm), DFloat] ≈ engf_eng0
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
