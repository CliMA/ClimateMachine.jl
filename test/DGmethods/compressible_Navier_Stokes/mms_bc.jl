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


struct AtmosModel <: BalanceLaw
end

dimension(::AtmosModel) = 3
vars_aux(::AtmosModel) = ()
vars_state(::AtmosModel) = (:ρ, :ρu, :ρv, :ρw, :ρe)
vars_state_for_transform(::AtmosModel) = (:ρ, :ρu, :ρv, :ρw)
vars_transform(::AtmosModel) = (:u, :v, :w)
vars_diffusive(::AtmosModel) = (:τ11, :τ22, :τ33, :τ12, :τ13, :τ23)

function flux!(::AtmosModel, flux::Grad, state::State, diffstate::State, auxstate::State, t::Real)
  # preflux
  γ = γ_exact  
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * state.ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)

  # invisc terms
  flux.ρ  = (state.ρu          , state.ρv          , state.ρw)
  flux.ρu = (u * state.ρu  + P , v * state.ρu      , w * state.ρu)
  flux.ρv = (u * state.ρv      , v * state.ρv + P  , w * state.ρv)
  flux.ρw = (u * state.ρw      , v * state.ρw      , w * state.ρw + P)
  flux.ρe = (u * (state.ρe + P), v * (state.ρe + P), w * (state.ρe + P))

  # viscous terms
  flux.ρu .-= (diffstate.τ11, diffstate.τ12, diffstate.τ13)
  flux.ρv .-= (diffstate.τ12, diffstate.τ22, diffstate.τ23)
  flux.ρw .-= (diffstate.τ13, diffstate.τ23, diffstate.τ33)

  flux.ρe .-= (u * diffstate.τ11 + v * diffstate.τ12 + w * diffstate.τ13,
               u * diffstate.τ12 + v * diffstate.τ22 + w * diffstate.τ23,
               u * diffstate.τ13 + v * diffstate.τ23 + w * diffstate.τ33)
end

function transform!(::AtmosModel, transformstate::State, state::State, auxstate::State, t::Real)
  ρinv = 1 / state.ρ
  transformstate.u = ρinv * state.ρu,
  transformstate.v = ρinv * state.ρv
  transformstate.w = ρinv * state.ρw
end

function diffusive!(::AtmosModel, diff::State, ∇transform::Grad)
  dudx, dudy, dudz = ∇transform.u
  dvdx, dvdy, dvdz = ∇transform.v
  dwdx, dwdy, dwdz = ∇transform.w

  # strains
  ϵ11 = dudx
  ϵ22 = dvdy
  ϵ33 = dwdz
  ϵ12 = (dudy + dvdx) / 2
  ϵ13 = (dudz + dwdx) / 2
  ϵ23 = (dvdz + dwdy) / 2

  # deviatoric stresses
  diff.τ11 = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diff.τ22 = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diff.τ33 = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
  diff.τ12 = 2μ * ϵ12
  diff.τ13 = 2μ * ϵ13
  diff.τ23 = 2μ * ϵ23
end

function source!(::AtmosModel, source::State, state::State, aux::State, t::Real)
  S.ρ  = Sρ_g(t, aux.x, aux.y, aux.z, Val(3))
  S.ρu = SU_g(t, aux.x, aux.y, aux.z, Val(3))
  S.ρv = SV_g(t, aux.x, aux.y, aux.z, Val(3))
  S.ρw = SW_g(t, aux.x, aux.y, aux.z, Val(3))
  S.ρe = SE_g(t, aux.x, aux.y, aux.z, Val(3))
end

function wavespeed(::AtmosModel, nM, state::State, aux::State, t::Real)
  γ = γ_exact
  ρinv = 1 / state.ρ
  u, v, w = ρinv * state.ρu, ρinv * ρv, ρinv * state.ρw
  P = (γ-1)*(state.ρe - ρinv * (state.ρu^2 + state.ρv^2 + state.ρw^2) / 2)

  return abs(n[1] * u + n[2] * v + n[3] * w) + sqrt(ρinv * γ * P)
end

function boundarycondition!(::AtmosModel, stateP::State, diffP::State, auxP::State, nM, stateM::State, diffM::State, auxM::State, bctype, t)
  initialcondition!(stateP, t, auxM.x, auxM.y, auxM.z)
end

struct MyGradNumFlux <: GradNumericalFlux
end

function diffusive_penalty!(::MyGradNumFlux, bl::BalanceLaw, VF, nM, velM, QM, aM, velP, QP, aP, t)
  @inbounds begin
    n_Δvel = similar(VF, Size(dimension(bl), vars_diffusive(bl))
    for j = 1:vars_diffusive(bl), i = 1:dimension(bl)
      n_Δvel[i, j] = nM[i] * (velP[j] - velM[j]) / 2
    end
    diffusive!(bl, State{vars_diffusive(bl)}(VF), Grad{vars_transform(bl)}(n_Δvel))
  end
end

@inline diffusive_boundary_penalty!(::MyGradNumFlux, bl::BalanceLaw, VF, _...) = VF.=0


if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

include("mms_solution_generated.jl")



# initial condition
function initialcondition!(dim, Q, t, x, y, z, _...)
  DFloat = eltype(Q)
  ρ::DFloat = ρ_g(t, x, y, z, dim)
  U::DFloat = U_g(t, x, y, z, dim)
  V::DFloat = V_g(t, x, y, z, dim)
  W::DFloat = W_g(t, x, y, z, dim)
  E::DFloat = E_g(t, x, y, z, dim)

  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E] = ρ, U, V, W, E
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
  errf
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
  expected_result = Array{Float64}(undef, 2, 3) # dim-1, lvl
  expected_result[1,1] = 1.6687745307357629e-01
  expected_result[1,2] = 5.4179126727473799e-03
  expected_result[1,3] = 2.3066157635992409e-04
  expected_result[2,1] = 3.3669188610024728e-02
  expected_result[2,2] = 1.7603468555920912e-03
  expected_result[2,3] = 9.1108572847298699e-05
  lvls = integration_testing ? size(expected_result, 2) : 1

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      result = zeros(DFloat, lvls)
      for dim = 2:3
        for l = 1:lvls
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
          timeend = 1
          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps

          @info (ArrayType, DFloat, dim)
          result[l] = run(mpicomm, ArrayType, dim, topl, warpfun,
                          polynomialorder, timeend, DFloat, dt)
          @test result[l] ≈ DFloat(expected_result[dim-1, l])
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
