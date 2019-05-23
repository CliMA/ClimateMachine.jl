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

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Vtk

using LinearAlgebra
using StaticArrays
using Printf

using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.Microphysics

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  const ArrayTypes = VERSION >= v"1.2-pre.25" ? (Array, CuArray) : (Array,)
else
  const ArrayTypes = (Array, )
end

const _nstate = 5
const _ρ, _U, _W, _E, _qt = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Wid = _W, Eid = _E, qtid = _qt)
const statenames = ("ρ", "U", "W", "E", "qt")

const _nauxcstate = 1
const _c_z = 1

# preflux computation for wavespeed function
@inline function preflux(Q, _...)

  @inbounds ρ, U, W = Q[_ρ], Q[_U], Q[_W]
  u, w = U / ρ, W / ρ
  (u, w)
end

# boundary condition
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t, uM, wM)
  @inbounds begin
    UM, WM, EM = QM[_U],  QM[_W],  QM[_E]
    qtM        = QM[_qt]

    UnM = nM[1] * UM + nM[2] * WM
    QP[_U] = UM - 2 * nM[1] * UnM
    QP[_W] = WM - 2 * nM[2] * UnM

    QP[_E], QP[_qt] = EM, qtM

    auxM .= auxP

    # Required return from this function is either nothing
    # or preflux with plus state as arguments
    preflux(QP, auxP, t)
  end
end


# max eigenvalue
@inline function wavespeed(n, Q, aux, t, u, w)
  @inbounds abs(n[1] * u + n[2] * w)
end


@inline function constant_auxiliary_init!(aux, x, z, _...)
  @inbounds aux[_c_z] = z
end


# physical flux function
eulerflux!(F, Q, QV, aux, t) = eulerflux!(F, Q, QV, aux, t, preflux(Q)...)

@inline function eulerflux!(F, Q, QV, aux, t, u, w)
  @inbounds begin
    E, qt = Q[_E], Q[_qt]

    F .= 0
    # advect the moisture and energy
    F[1, _qt], F[2, _qt] = u * qt, w * qt
    F[1, _E],  F[2, _E]  = u *  E, w * E
    # don't advect momentum (kinematic setup)
  end
end


# initial condition
const w_max = .6    # m/s
const Z_max = 1500. # m
const X_max = 1500. # m

@inline function single_eddy!(Q, t, x, z, _...)
  DFloat = eltype(Q)

  # initial condition
  θ_0::DFloat    = 289         # K
  p_0::DFloat    = 101500      # Pa
  p_1000::DFloat = 100000      # Pa
  qt_0::DFloat   = 7.5 * 1e-3  # kg/kg
  z_0::DFloat    = 0           # m

  R_m, cp_m, cv_m, γ = moist_gas_constants(PhasePartition(qt_0))

  # Pressure profile assuming hydrostatic and constant θ and qt profiles.
  # It is done this way to be consistent with Arabas paper.
  # It's not neccesarily the best way to initialize with our model variables.
  p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
              R_d / cp_d * grav / θ_0 / R_m * (z - z_0)
             )^(cp_d / R_d)

  T::DFloat = θ_0 * exner(p, PhasePartition(qt_0))
  ρ::DFloat = p / R_m / T

  pp_init = PhasePartition_equil(T, ρ, qt_0)
  thermo_state_init = PhaseEquil(internal_energy(T, pp_init), qt_0, ρ)
  qt::DFloat = qt_0

  # TODO should this be more "grid aware"?
  # the velocity is calculated as derivative of streamfunction
  U::DFloat = w_max * X_max/Z_max * cos(π * z/Z_max) * cos(2*π * x/X_max)
  W::DFloat = 2*w_max * sin(π * z/Z_max) * sin(2*π * x/X_max)

  u = U/ρ
  w = W/ρ

  E = ρ * (grav * z + (1//2)*(u^2 + w^2) + internal_energy(thermo_state_init))

  @inbounds Q[_ρ], Q[_U], Q[_W], Q[_E], Q[_qt] = ρ, U, W, E, qt
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
                             constant_auxiliary_init!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = single_eddy!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  npoststates = 2
  _ql, _qi= 1:npoststates
  postnames = ("ql", "qi")
  postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

  writevtk("initial_condition", Q, spacedisc, statenames)

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

  cbvtk = GenericCallbacks.EveryXSimulationSteps(10) do (init=false)

    DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,
                                               Q) do R, Q, QV, aux
      @inbounds let
        ρ, E, U, W, qt = Q[_ρ], Q[_E], Q[_U], Q[_W], Q[_qt]
        z = aux[_c_z]

        e_int = (E - 1//2 * (U^2 + W^2) - grav * z) / ρ
        ts  =  PhaseEquil(e_int, qt, ρ)  # saturation adjustment happens here

        R[_ql] = PhasePartition(ts).liq
        R[_qi] = PhasePartition(ts).ice
      end
    end

    outprefix = @sprintf("vtk/eddy_sat_adj_%dD_mpirank%04d_step%04d",
                         dim, MPI.Comm_rank(mpicomm), step[1])
    @printf(io, "----\n")
    @printf(io, "doing VTK output =  %s\n", outprefix)
    writevtk(outprefix, Q, spacedisc, statenames, postprocessarray, postnames)
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  Qe = MPIStateArray(spacedisc,
                    (Q, x...) -> single_eddy!(Q, DFloat(timeend), x...))

  # Print some end of the simulation information
  engf = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ ( final ) = %.16e\n", engf)
end

function run(dim, Ne, N, timeend, DFloat)

  #ArrayType = CuArray
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=Z_max), 2)

  topl = BrickTopology(mpicomm, brickrange, periodicity=(true, false))
  dt = 1.

  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)

end

using Test
let
  timeend = 2
  numelem = (75, 75)
  lvls = 3
  dim = 2
  DFloat = Float64

  polynomialorder = 4

  run(dim, ntuple(j->numelem[j], dim), polynomialorder, timeend, DFloat)
end

isinteractive() || MPI.Finalize()

nothing
