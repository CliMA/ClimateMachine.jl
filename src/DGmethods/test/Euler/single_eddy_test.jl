# The test is based on a modelling set-up designed for the
# 8th International Cloud Modelling Workshop
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See chapter 2 in Arabas et al 2015 for setup details:
#@Article{gmd-8-1677-2015,
#AUTHOR = {Arabas, S. and Jaruga, A. and Pawlowska, H. and Grabowski, W. W.},
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk, and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}

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
using LinearAlgebra
using StaticArrays
using Printf

using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.Microphysics

const _nstate = 8
const _ρ, _U, _W, _E, _qt, _ql, _qi, _qr = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Wid = _W, Eid = _E,
                 qtid = _qt, qlid = _ql, qiid = _qi, qrid = _qr)
const statenames = ("ρ", "U", "W", "E", "qt", "ql", "qi", "qr")

const _nauxcstate = 2
const _c_z = 1
const _c_x = 2


# preflux computation
@inline function preflux(Q, _...)
  DFloat = eltype(Q)

  @inbounds ρ, U, W, qt, qr = Q[_ρ], Q[_U], Q[_W], Q[_qt], Q[_qr]

  ρinv = 1 / ρ
  ρ_ground::DFloat = 1 #TODO ρ[0]

  u, w = ρinv * U, ρinv * W
  rain_w = terminal_velocity(qt, qr, ρ, ρ_ground)
  rain_w = 0

  # return 2 velocity components and rain fall speed for wave speed calculation
  (u, w, rain_w)
end


# boundary condition
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t,
                          uM, wM, rain_wM)
  @inbounds begin

    UM, WM, EM         = QM[_U],  QM[_W],  QM[_E]
    qtM, qlM, qiM, qrM = QM[_qt], QM[_ql], QM[_qi], QM[_qr]

    UnM = nM[1] * UM + nM[2] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_W] = WM - 2 * nM[2] * UnM # TODO - what to do about rain fall speed?

    QP[_E], QP[_qt], QP[_ql], QP[_qi], QP[_qr] = EM, qtM, qlM, qiM, qrM

    auxM .= auxP

    # To calculate uP and wP we use the preflux function
    preflux(QP, auxP, t)

    # Required return from this function is either nothing
    # or preflux with plus state as arguments
  end
end


# max eigenvalue
@inline function wavespeed(n, Q, aux, t, u, w, rain_w)
  @inbounds abs(n[1] * u + n[2] * max(w, rain_w, w+rain_w))
end


@inline function constant_auxiliary_init!(aux, x, z, _...)
  @inbounds aux[_c_z] = z
  @inbounds aux[_c_x] = x #TODO - tmp for printing
end


@inline function source!(S, Q, aux, t)
  @inbounds begin
    ρ, E, U, W, qt, ql, qi = Q[_ρ], Q[_E], Q[_U], Q[_W], Q[_qt], Q[_ql], Q[_qi]
    z = aux[_c_z]
    x = aux[_c_x]

    timescale::eltype(Q) = 1

    e_int = (E - 1//2 * (U^2 + W^2) - grav * z) / ρ

    # TODO

    # TODO pp created every time I calculate sources...
    #pp = PhasePartition(qt, ql)

    #TODO - add also the equilibrum case example
    #T = saturation_adjustment(e_int, ρ, qt)
    #T = air_temperature(e_int, pp)

    #dqldt, dqidt = qv2qli(pp, T, ρ, x, z, timescale)

    #dqrdt = ql2qr(pp, timescale)
    #dqrdt = 0

    #if x == 0
    #  @printf("z = %5.5f  dqrdt =  %5.5f\n ", z, dqrdt)
    #  @printf("  ")
    #end

    S .= 0

    #if x == 0
    #  @printf("z = %5.5f  T = %5.3f  qt =  %5.4f  ql = %5.8f\n", z, T, qt, ql)
    #end

    # TODO
    #S[_ql] = dqldt
    #S[_ql] = dqldt - dqrdt

    #S[_qi] = dqidt

    #S[_qr] = dqrdt
    #S[_qt] = -dqrdt

    # TODO add src to E

  end
end


# physical flux function
eulerflux!(F, Q, QV, aux, t) = eulerflux!(F, Q, QV, aux, t, preflux(Q)...)

@inline function eulerflux!(F, Q, QV, aux, t, u, w, rain_w)
  @inbounds begin
    E, qt, ql, qi, qr = Q[_E], Q[_qt], Q[_ql], Q[_qi], Q[_qr]

    F .= 0
    # advect the moisture and energy
    F[1, _qt], F[2, _qt] = u * qt, w * qt
    F[1, _ql], F[2, _ql] = u * ql, w * ql
    F[1, _qi], F[2, _qi] = u * qi, w * qi
    F[1, _qr], F[2, _qr] = u * qr, (w + rain_w) * qr
    F[1, _E],  F[2, _E]  = u *  E, w * E
    # don't advect momentum (kinematic setup)
  end
end


# initial condition
const w_max = .6
const Z_max = 1.5
const X_max = 1.5

function single_eddy!(Q, t, x, z, _...)
  DFloat = eltype(Q)

  # initial condition
  θ_0::DFloat    = 289
  p_0::DFloat    = 101500
  p_1000::DFloat = 100000
  qt_0::DFloat   = 7.5 * 1e-3
  z_0::DFloat    = 0

  R_m, cp_m, cv_m, γ = moist_gas_constants(PhasePartition(qt_0))

  # Pressure profile assuming hydrostatic and constant θ and qt profiles.
  # It is done this way to be consistent with Arabas paper.
  # It's not neccesarily the best way to initialize with our model variables.
  p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
              R_d / cp_d * grav / θ_0 / R_m * (z - z_0) * 1e3
             )^(cp_d / R_d)

  T::DFloat = θ_0 * exner(p, PhasePartition(qt_0))
  ρ::DFloat = p / R_m / T

  pp_init = PhasePartition_equil(T, ρ, qt_0)
  thermo_state_init = PhaseEquil(internal_energy(T, pp_init), qt_0, ρ)
  qt::DFloat = qt_0
  ql::DFloat = pp_init.liq
  qi::DFloat = pp_init.ice
  qr::DFloat = 0

  # TODO should this be more "grid aware"?
  # the velocity is calculated as derivative of streamfunction
  U::DFloat = w_max * X_max/Z_max * cos(π * z/Z_max) * cos(2*π * x/X_max)
  W::DFloat = 2*w_max * sin(π * z/Z_max) * sin(2*π * x/X_max)

  u = U/ρ
  w = W/ρ

  E = ρ * (grav * z + (1//2)*(u^2 + w^2) + internal_energy(thermo_state_init))

  @inbounds Q[_ρ], Q[_U], Q[_W], Q[_E], Q[_qt], Q[_ql], Q[_qi], Q[_qr] =
            ρ, U, W, E, qt, ql, qi, qr
end

function main(mpicomm, DFloat, topl::AbstractTopology{dim}, N, timeend,
              ArrayType, dt) where {dim}

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  numflux!(x...) = NumericalFluxes.rusanov!(x...,
                                            eulerflux!,
                                            wavespeed,
                                            preflux
                                           )
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x...,
                                                            eulerflux!,
                                                            bcstate!,
                                                            wavespeed,
                                                            preflux
                                                           )

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = eulerflux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!,
                           auxiliary_state_length = _nauxcstate,
                           auxiliary_state_initialization! =
                             constant_auxiliary_init!,
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

  topl = BrickTopology(mpicomm, brickrange, periodicity=(true, false))
  dt = 0.001 # TODO not a general purpose dt calculation

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
