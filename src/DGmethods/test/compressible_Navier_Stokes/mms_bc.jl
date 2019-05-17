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

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  const ArrayTypes = VERSION >= v"1.2-pre.25" ? (Array, CuArray) : (Array,)
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

# flux function
cns_flux!(F, Q, VF, aux, t) = cns_flux!(F, Q, VF, aux, t, preflux(Q)...)

@inline function cns_flux!(F, Q, VF, aux, t, P, u, v, w, ρinv)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    τ11, τ22, τ33 = VF[_τ11], VF[_τ22], VF[_τ33]
    τ12 = τ21 = VF[_τ12]
    τ13 = τ31 = VF[_τ13]
    τ23 = τ32 = VF[_τ23]

    # inviscid terms
    F[1, _ρ], F[2, _ρ], F[3, _ρ] = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U] = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V] = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W] = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E] = u * (E + P), v * (E + P), w * (E + P)

    # viscous terms
    F[1, _U] -= τ11; F[2, _U] -= τ12; F[3, _U] -= τ13
    F[1, _V] -= τ21; F[2, _V] -= τ22; F[3, _V] -= τ23
    F[1, _W] -= τ31; F[2, _W] -= τ32; F[3, _W] -= τ33

    F[1, _E] -= u * τ11 + v * τ12 + w * τ13
    F[2, _E] -= u * τ21 + v * τ22 + w * τ23
    F[3, _E] -= u * τ31 + v * τ32 + w * τ33
  end
end

# Compute the velocity from the state
@inline function velocities!(vel, Q, _...)
  @inbounds begin
    # ordering should match states_for_gradient_transform
    ρ, U, V, W = Q[1], Q[2], Q[3], Q[4]
    ρinv = 1 / ρ
    vel[1], vel[2], vel[3] = ρinv * U, ρinv * V, ρinv * W
  end
end

# Visous flux
@inline function compute_stresses!(VF, grad_vel, _...)
  μ::eltype(VF) = μ_exact
  @inbounds begin
    dudx, dudy, dudz = grad_vel[1, 1], grad_vel[2, 1], grad_vel[3, 1]
    dvdx, dvdy, dvdz = grad_vel[1, 2], grad_vel[2, 2], grad_vel[3, 2]
    dwdx, dwdy, dwdz = grad_vel[1, 3], grad_vel[2, 3], grad_vel[3, 3]

    # strains
    ϵ11 = dudx
    ϵ22 = dvdy
    ϵ33 = dwdz
    ϵ12 = (dudy + dvdx) / 2
    ϵ13 = (dudz + dwdx) / 2
    ϵ23 = (dvdz + dwdy) / 2

    # deviatoric stresses
    VF[_τ11] = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ22] = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ33] = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
    VF[_τ12] = 2μ * ϵ12
    VF[_τ13] = 2μ * ϵ13
    VF[_τ23] = 2μ * ϵ23
  end
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
    DGBalanceLawDiscretizations.writevtk(outprefix, Q, spacedisc, statenames,
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
  @static if Base.find_package("CUDAnative") !== nothing
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
