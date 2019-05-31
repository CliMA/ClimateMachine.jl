using MPI
using CLIMA
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
using CLIMA.Vtk

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

const _nviscstates = 6
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23 = 1:_nviscstates

const _ngradstates = 3
const _states_for_gradient_transform = (_ρ, _U, _V, _W)

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
  using Random
end

include("mms_solution_generated.jl")

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

struct State{T}
  arr::T
end
function Base.getproperty(state::State, sym::Symbol)
  arr = getfield(state,:arr)
  if sym == :ρ
    @inbounds arr[_ρ]
  elseif sym == :ρ𝐮
    @inbounds SVector(arr[_U], arr[_V], arr[_W])
  elseif sym == :𝐮
    @inbounds SVector(arr[_U], arr[_V], arr[_W]) ./ arr[_ρ]
  elseif sym == :ρe
    @inbounds arr[_E]
  else
    error("no property $sym")
  end
end

struct Flux{T}
  arr::T
end

function Base.setproperty!(flux::Flux, sym::Symbol, val)
  arr = getfield(flux,:arr)
  if sym == :ρ
    @inbounds arr[:,_ρ] = val
  elseif sym == :ρ𝐮
    @inbounds arr[:,_U:_W] = val
  elseif sym == :ρe
    @inbounds arr[:,_E] = val
  else
    error("no property $sym")
  end
end

struct Stress{T}
  arr::T
end
function Base.getproperty(stress::Stress, sym::Symbol)
  arr = getfield(stress,:arr)
  if sym == :𝛕
    @inbounds SMatrix{3,3}(arr[_τ11], arr[_τ12], arr[_τ13],
                           arr[_τ12], arr[_τ22], arr[_τ23],
                           arr[_τ13], arr[_τ23], arr[_τ33])
  else
    error("no property $sym")
  end
end
function Base.setproperty!(stress::Stress, sym::Symbol, val)
  arr = getfield(stress,:arr)
  if sym == :𝛕    
    @inbounds begin
      arr[_τ11] = val[1,1]
      arr[_τ22] = val[2,2]
      arr[_τ33] = val[3,3]
      arr[_τ12] = val[1,2]
      arr[_τ13] = val[1,3]
      arr[_τ23] = val[2,3]
    end
  else
    error("no property $sym")
  end
end


# flux function
cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q)...)

@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv)
  flux = Flux(F)
  state = State(Q)
  stress = Stress(VF)

  flux.ρ  = state.ρ𝐮
  flux.ρ𝐮 = state.𝐮 * state.ρ𝐮' + P*I - stress.𝛕
  flux.ρe  = (state.ρe + P) * state.𝐮 - stress.𝛕 * state.𝐮
end

# Compute the velocity from the state
@inline function velocities!(𝐮, Q, _...)
  state = State(Q)
  𝐮 .= state.𝐮
end

# Visous flux
@inline function compute_stresses!(VF, ∇𝐮, _...)
  μ::eltype(VF) = μ_exact
  stress = Stress(VF)
  stress.𝛕 = μ * ((∇𝐮 + ∇𝐮') - 2*tr(∇𝐮)/3*I)
end

@inline function stresses_penalty!(VF, nM, velM, QM, aM, velP, QP, aP, t)
  @inbounds begin
    n_Δvel = similar(VF, Size(3, 3))
    for j = 1:3, i = 1:3
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    compute_stresses!(VF, n_Δvel)
  end
end

@inline stresses_boundary_penalty!(VF, _...) = VF.=0

# initial condition
function initialcondition!(dim, Q, t, x, y, z, _...)
  DFloat = eltype(Q)
  ρ::DFloat = ρ_g(t, x, y, z, dim)
  U::DFloat = U_g(t, x, y, z, dim)
  V::DFloat = V_g(t, x, y, z, dim)
  W::DFloat = W_g(t, x, y, z, dim)
  E::DFloat = E_g(t, x, y, z, dim)

  if integration_testing
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
  else
    @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] =
    10+rand(), rand(), rand(), rand(), 10+rand()
  end
end

const _nauxstate = 3
const _a_x, _a_y, _a_z = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
  end
end

@inline function source3D!(S, Q, aux, t)
  @inbounds begin
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    S[_ρ] = Sρ_g(t, x, y, z, Val(3))
    S[_U] = SU_g(t, x, y, z, Val(3))
    S[_V] = SV_g(t, x, y, z, Val(3))
    S[_W] = SW_g(t, x, y, z, Val(3))
    S[_E] = SE_g(t, x, y, z, Val(3))
  end
end

@inline function source2D!(S, Q, aux, t)
  @inbounds begin
    x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
    S[_ρ] = Sρ_g(t, x, y, z, Val(2))
    S[_U] = SU_g(t, x, y, z, Val(2))
    S[_V] = SV_g(t, x, y, z, Val(2))
    S[_W] = SW_g(t, x, y, z, Val(2))
    S[_E] = SE_g(t, x, y, z, Val(2))
  end
end

@inline function bcstate2D!(QP, QVP, auxP, nM, QM, QVM, auxM, bctype, t, _...)
  @inbounds begin
    x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
    if integration_testing
      initialcondition!(Val(2), QP, t, x, y, z)
    else
      for s = 1:length(QP)
        QP[s] = QM[length(QP)+1-s]
      end
      for s = 1:_nviscstates
        QVP[s] = QVM[s]
      end
    end
  end
  nothing
end

@inline function bcstate3D!(QP, QVP, auxP, nM, QM, QVM, auxM, bctype, t, _...)
  @inbounds begin
    x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
    if integration_testing
      initialcondition!(Val(3), QP, t, x, y, z)
    else
      for s = 1:length(QP)
        QP[s] = QM[length(QP)+1-s]
      end
      for s = 1:_nviscstates
        QVP[s] = QVM[s]
      end
    end
  end
  nothing
end

function run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, DFloat, dt)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                          meshwarp = warpfun,
                                         )

  # spacedisc = data needed for evaluating the right-hand side function
  numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed,
                                            preflux)
  bcstate! = dim == 2 ?  bcstate2D! : bcstate3D!
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!,
                                                            bcstate!,
                                                            wavespeed, preflux)
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = cns_flux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!,
                           number_gradient_states = _ngradstates,
                           states_for_gradient_transform =
                             _states_for_gradient_transform,
                           number_viscous_states = _nviscstates,
                           gradient_transform! = velocities!,
                           viscous_transform! = compute_stresses!,
                           viscous_penalty! = stresses_penalty!,
                           viscous_boundary_penalty! =
                           stresses_boundary_penalty!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           auxiliary_state_initialization!,
                           source! = dim == 2 ? source2D! : source3D!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = initialcondition!(Val(dim), Q, DFloat(0), x...)
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
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  npoststates = 5
  _P, _u, _v, _w, _ρinv = 1:npoststates
  postnames = ("P", "u", "v", "w", "ρinv")
  postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

  step = [0]
  mkpath("vtk")
  cbvtk = GenericCallbacks.EveryXSimulationSteps(100) do (init=false)
    DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,
                                               Q) do R, Q, QV, aux
      @inbounds let
        (R[_P], R[_u], R[_v], R[_w], R[_ρinv]) = preflux(Q)
      end
    end

    outprefix = @sprintf("vtk/cns_%dD_mpirank%04d_step%04d", dim,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, spacedisc, statenames,
             postprocessarray, postnames)
    step[1] += 1
    nothing
  end

  # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, ))
  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  if integration_testing
    Qe = MPIStateArray(spacedisc,
                       (Q, x...) -> initialcondition!(Val(dim), Q,
                                                      DFloat(timeend), x...))
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

using Test
let
  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())
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

  polynomialorder = 4
  base_num_elem = 4
  if integration_testing
    expected_result = Array{Float64}(undef, 2, 3) # dim-1, lvl
    expected_result[1,1] = 1.6687745307357629e-01
    expected_result[1,2] = 5.4179126727473799e-03
    expected_result[1,3] = 2.3066157635992409e-04
    expected_result[2,1] = 3.3669188610024728e-02
    expected_result[2,2] = 1.7603468555920912e-03
    expected_result[2,3] = 9.1108572847298699e-05
    lvls = size(expected_result, 2)
  else
    expected_result = Dict{Tuple{Int64, Int64, DataType}, AbstractFloat}()
    expected_result[2, 1, Float64] = 9.9075897196717488e-01
    expected_result[3, 1, Float64] = 1.0099522817205739e+00
    expected_result[2, 3, Float64] = 9.9072475063319887e-01
    expected_result[3, 3, Float64] = 1.0099521111150005e+00
    lvls = 1
  end


  for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      result = zeros(DFloat, lvls)
      for dim = 2:3
        for l = 1:lvls
          integration_testing || Random.seed!(0)
          if dim == 2
            Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
            brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                          range(DFloat(0); length=Ne[2]+1, stop=1))
            topl = BrickTopology(mpicomm, brickrange,
                                 periodicity = (false, false))
            dt = 1e-2 / Ne[1]
            warpfun = (x, y, _) -> begin
              (x + sin(x*y), y + sin(2*x*y), 0)
            end

          elseif dim == 3
            Ne = (2^(l-1) * base_num_elem, 2^(l-1) * base_num_elem)
            brickrange = (range(DFloat(0); length=Ne[1]+1, stop=1),
                          range(DFloat(0); length=Ne[2]+1, stop=1),
            range(DFloat(0); length=Ne[2]+1, stop=1))
            topl = BrickTopology(mpicomm, brickrange,
                                 periodicity = (false, false, false))
            dt = 5e-3 / Ne[1]
            warpfun = (x, y, z) -> begin
              (x + (x-1/2)*cos(2*π*y*z)/4,
               y + exp(sin(2π*(x*y+z)))/20,
              z + x/4 + y^2/2 + sin(x*y*z))
            end
          end
          timeend = integration_testing ? 1 : 2dt
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps

          @info (ArrayType, DFloat, dim)
          result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                          polynomialorder, timeend, DFloat, dt)
          if integration_testing
            @test result[l] ≈ DFloat(expected_result[dim-1, l])
          else
            @test result[l] ≈ expected_result[dim, MPI.Comm_size(mpicomm), DFloat]
          end
        end
        if integration_testing
          @info begin
            msg = ""
            for l = 1:lvls-1
              rate = log2(result[l]) - log2(result[l+1])
              msg *= @sprintf("\n  rate for level %d = %e\n", l, rate)
            end
            msg
          end
        end
      end
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
