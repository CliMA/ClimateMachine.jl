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
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.VTK

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E)
const statenames = ("ρ", "U", "V", "W", "E")
const γ_exact = 7 // 5
if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# preflux computation
@inline function preflux(Q)
  γ::eltype(Q) = γ_exact
  @inbounds ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
  ρinv = 1 / ρ
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  ((γ-1)*(E - ρinv * (U^2 + V^2 + W^2) / 2), u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t)
  P, u, v, w, ρinv = preflux(Q)
  γ::eltype(Q) = γ_exact
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

# physical flux function
@inline function eulerflux!(F, Q, QV, aux, t)
  P, u, v, w, ρinv = preflux(Q)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)
  end
end

const _nauxstate = 3
const _a_intρ, _a_intρ2, _a_intρ_reverse = 1:_nauxstate

# initial condition
const halfperiod = 5
function isentropicvortex!(Q, t, x, y, z, _...)
  FT = eltype(Q)

  γ::FT    = γ_exact
  uinf::FT = 2
  vinf::FT = 1
  Tinf::FT = 1
  λ::FT    = 5

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
  w = zero(FT)

  ρ = (Tinf - ((γ-1)*λ^2*exp(2*(1-rsq))/(γ*16*π*π)))^(1/(γ-1))
  p = ρ^γ
  U = ρ*u
  V = ρ*v
  W = ρ*w
  E = p/(γ-1) + (1//2)*ρ*(u^2 + v^2 + w^2)

  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E

end

@inline function integral_knl(val, Q, aux)
  @inbounds begin
    ρ = Q[_ρ]
    val[1] = ρ
    val[2] = ρ^2
  end
end
function integral_computation(disc, Q, t)
  DGBalanceLawDiscretizations.indefinite_stack_integral!(disc, integral_knl, Q,
                                                         (_a_intρ, _a_intρ2))
  DGBalanceLawDiscretizations.reverse_indefinite_stack_integral!(disc,
                                                                 _a_intρ_reverse,
                                                                 _a_intρ)
end

function main(mpicomm, FT, topl::AbstractTopology{dim}, N, timeend,
              ArrayType, dt) where {dim}

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = FT,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  numflux!(x...) = NumericalFluxes.rusanov!(x..., eulerflux!, wavespeed)
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = eulerflux!,
                           numerical_flux! = numflux!,
                           auxiliary_state_length = _nauxstate,
                           preodefun! = integral_computation,
                          )

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = isentropicvortex!(Q, 0, x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  # Compute the initial integral (which will be compared with final integral)
  integral_computation(spacedisc, Q, 0)
  exact_aux = copy(spacedisc.auxstate)

  lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

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
    writevtk(outprefix, Q, spacedisc, statenames)
    step[1] += 1
    nothing
  end

  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  Qe = MPIStateArray(spacedisc,
                     (Q, x...) -> isentropicvortex!(Q, timeend, x...))
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  err_int = euclidean_distance(spacedisc.auxstate, exact_aux)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  err_int                 = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe err_int
  (errf, err_int)
end

function run(mpicomm, ArrayType, dim, Ne, N, timeend, FT, dt)
  brickrange = ntuple(j->range(FT(-halfperiod); length=Ne[j]+1,
                               stop=halfperiod), dim)
  topl = StackedBrickTopology(mpicomm, brickrange,
                              periodicity=ntuple(j->true, dim))
  main(mpicomm, FT, topl, N, timeend, ArrayType, dt)
end

using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  timeend = 2halfperiod
  numelem = (5, 5, 1)

  polynomialorder = 4

  expected_error = Array{Float64}(undef, 2, 3) # dim-1, lvl
  expected_error[1,1] = 1.0948999661850387e+00
  expected_error[1,2] = 2.0267925216657484e-01
  expected_error[1,3] = 5.5082585722351137e-03
  expected_error[2,1] = 3.4623777032091780e+00
  expected_error[2,2] = 6.4092807130049356e-01
  expected_error[2,3] = 1.7418643028977332e-02
  expected_integral_error = Array{Float64}(undef, 2, 3) # dim-1, lvl
  expected_integral_error[1,1] = 4.8681695495452398e-01
  expected_integral_error[1,2] = 8.8617915194155406e-02
  expected_integral_error[1,3] = 2.3262199146720022e-03
  expected_integral_error[2,1] = 5.2924082817426514e+00
  expected_integral_error[2,2] = 1.1950747332395113e+00
  expected_integral_error[2,3] = 4.2466533227878363e-02
  lvls = integration_testing ? size(expected_error, 2) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for FT in (Float64,) #Float32)
      for dim = 2:3
        err = zeros(FT, lvls)
        int_err = zeros(FT, lvls)
        for l = 1:lvls
          Ne = ntuple(j->2^(l-1) * numelem[j], dim)
          dt = 1e-2 / Ne[1]
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps
          @info (ArrayType, FT, dim)
          (err[l], int_err[l]) = run(mpicomm, ArrayType, dim, Ne,
                                     polynomialorder, timeend, FT, dt)
          @test err[l] ≈ FT(expected_error[dim-1, l])
          @test int_err[l] ≈ FT(expected_integral_error[dim-1, l])
        end
        @info begin
          msg = "solution rates"
          for l = 1:lvls-1
            rate = log2(err[l]) - log2(err[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
        @info begin
          msg = "integral rates"
          for l = 1:lvls-1
            rate = log2(int_err[l]) - log2(int_err[l+1])
            msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
          end
          msg
        end
      end
    end
  end
end

nothing
