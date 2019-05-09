# The test is based on a modelling set-up designed for the 
# 8th International Cloud Modelling Workshop 
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See Arabas et al 2015 chapter 2 (doi:10.5194/gmd-8-1677-2015)
# for a detailed description of setup and results.
#
# TODO - add some comparison with reference profiles from there?

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

const _nstate = 8
const _ρ, _U, _W, _E, _qt, _ql, _qi, _qr = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Wid = _W, Eid = _E,
                 qtid = _qt, qlid = _ql, qiid = _qi, qrid = _qr)
const statenames = ("ρ", "U", "W", "E", "qt", "ql", "qi", "qr")

const _nauxcstate = 1
const _c_z = 1

using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.Microphysics

# preflux computation
@inline function preflux(Q, _...)
  DFloat = eltype(Q)

  @inbounds ρ, U, W, qt, qr = Q[_ρ], Q[_U], Q[_W], Q[_qt], Q[_qr]

  ρinv = 1 / ρ
  ρ_ground::DFloat = 1 #TODO ρ[0]

  u, w = ρinv * U, ρinv * W
  rain_w = terminal_velocity(qt, qr, ρ, ρ_ground)
  rain_w = 0

  (u, w, rain_w)
end

# max eigenvalue
#TODO - plus rain terminal velocity?
#TODO - removed sound_speed - velocity is prescribed and density is const
#TODO - arguments...
@inline function wavespeed(n, Q, G, ϕ_c, ϕ_d, t, u, w, rain_w)
  @inbounds abs(n[1] * u + n[2] * max(w, rain_w, w+rain_w))
end

@inline function correctQ!(Q, _...)
  @inbounds Q[_ρ] = Q[_U] = Q[_W] = 0
end

@inline function constant_auxiliary_init!(auxc, x, z, _...)
  @inbounds auxc[_c_z] = z
end

@inline function source!(S, Q, G, auxc, auxd, t)
  @inbounds begin
    ρ, E, U, W, qt, ql, qi = Q[_ρ], Q[_E], Q[_U], Q[_W], Q[_qt], Q[_ql], Q[_qi]
    z = auxc[_c_z]

    timescale::eltype(Q) = 1

    e_int = (E - 1//2 * (U^2 + W^2) - grav * z) / ρ
    
    e_int > 0 || @show e_int, ρ, qt
    T = saturation_adjustment(e_int, ρ, qt)
    dqldt, dqidt = qv2qli(qt, ql, qi, T, ρ, timescale)
    dqrdt = ql2qr(ql, timescale, 1e-8)
    S .= 0
    #S[_ql], S[_qi] = dqldt, dqidt  #TODO add src to E and ql
    S[_ql], S[_qi], S[_qr], S[_qt] = dqldt - dqrdt, dqidt, dqrdt, -dqrdt #TODO add src to E and ql

end
end

# physical flux function
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t) =
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, preflux(Q)...)

@inline function eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, u, w, rain_w)
  @inbounds begin
    E, qt, ql, qi, qr = Q[_E], Q[_qt], Q[_ql], Q[_qi], Q[_qr]

    F .= 0
    F[1, _qt], F[2, _qt] = u * qt, w * qt
    F[1, _ql], F[2, _ql] = u * ql, w * ql
    F[1, _qi], F[2, _qi] = u * qi, w * qi
    F[1, _qr], F[2, _qr] = u * qr, (w + rain_w) * qr
    F[1, _E],  F[2, _E]  = u *  E, w * E
  end
end

# initial condition
const w_max = .6
const Z_max = 1.5
const X_max = 1.5

function single_eddy!(Q, t, x, z, _...)
  DFloat = eltype(Q)

  θ_0::DFloat  = 289
  p_0::DFloat  = 101500
  qt_0::DFloat = 15 * 1e-3
  z_0::DFloat  = 0

  R_m, cp_m, cv_m, γ = moist_gas_constants(qt_0)

  # pressure profile assuming hydrostatic and constant θ and qt profiles
  # TODO - check
  p = MSLP * ((p_0 / MSLP)^(R_m / cp_m) -
              R_m / cp_m * grav / θ_0 / R_m * (z - z_0)
             )^(cp_m / R_m)

  T::DFloat = θ_0 * exner(p)
  ρ::DFloat = p / R_m / T

  qt::DFloat = qt_0
  ql::DFloat, qi::DFloat = phase_partitioning_eq(T, ρ, qt)
  qr::DFloat = 0


  # TODO should this be more "grid aware"?
  # the velocity is calculated as derivative of streamfunction
  U::DFloat = w_max * X_max/Z_max * cos(π * z/Z_max) * cos(2*π * x/X_max)
  W::DFloat = 2*w_max * sin(π * z/Z_max) * sin(2*π * x/X_max)

  u = U/ρ
  w = W/ρ

  E = ρ * (grav * z + (1//2)*(u^2 + w^2) + internal_energy(T, qt))

  @inbounds Q[_ρ], Q[_U], Q[_W], Q[_E], Q[_qt], Q[_ql], Q[_qi], Q[_qr] =
            ρ, U, W, E, qt, ql, qi, qr

  #TODO - plot velocity and E and qt fields
  #TODO - should I do saturation adjustemnt in my init cond?
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
                           NumericalFluxes.rusanov!(x..., eulerflux!,
                                                    wavespeed,
                                                    preflux,
                                                    correctQ!
                                                   ),
                           length_constant_auxiliary = _nauxcstate,
                           constant_auxiliary_init! = constant_auxiliary_init!,
                           source! = source!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = single_eddy!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)
  DGBalanceLawDiscretizations.writevtk("initial_condition", Q, spacedisc, statenames)

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
  cbvtk = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
    outprefix = @sprintf("vtk/single_eddy_source_%dD_mpirank%04d_step%04d",
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
                    (Q, x...) -> single_eddy!(Q, DFloat(timeend), x...))

  # Print some end of the simulation information
  engf = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ ( final ) = %.16e\n", engf)
end

function run(dim, Ne, N, timeend, DFloat)
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=1.5), 2)

  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(i->true, dim))
  dt = 0.001 # not a general purpose dt calculation

  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
  
end

using Test
let
  timeend = 10
  numelem = (75, 75)
  lvls = 3
  dim = 2
  DFloat = Float64

  polynomialorder = 4

  run(dim, ntuple(j->numelem[j], dim), polynomialorder, timeend, DFloat)
end

isinteractive() || MPI.Finalize()

nothing
