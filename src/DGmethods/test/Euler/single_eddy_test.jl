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

const _nstate = 9
const _ρ, _U, _V, _W, _E, _qt, _ql, _qi, _qr = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, 
                 qtid = _qt, qlid = _ql, qiid = _qi, qrid = _qr)
const statenames = ("ρ", "U", "V", "W", "E", "qt", "ql", "qi", "qr")

using CLIMA.PlanetParameters: grav, MSLP
using CLIMA.MoistThermodynamics: internal_energy, moist_gas_constants, exner
using CLIMA.Microphysics: qv2qli, terminal_velocity

# preflux computation
@inline function preflux(Q, _...)
  DFloat = eltype(Q)

  @inbounds ρ, U, V, W, qt, qr = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_qt], Q[_qr]

  ρinv = 1 / ρ
  ρ_ground = 1 #TODO ρ[0]

  u, v, w = ρinv * U, ρinv * V, ρinv * W
  rain_w = terminal_velocity(qt, qr, ρ, ρ_ground)

  (u, v, w, rain_w)
end

# max eigenvalue
#TODO - plus rain terminal velocity?
#TODO - removed sound_speed - velocity is prescribed and density is const
#TODO - arguments...
@inline function wavespeed(n, Q, G, ϕ_c, ϕ_d, t, P, u, v, w, T)
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w)
end

# physical flux function
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t) =
eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, preflux(Q)...)

@inline function eulerflux!(F, Q, G, ϕ_c, ϕ_d, t, P, u, v, w, T)
  @inbounds begin
    E, qt, ql, qi, qr = Q[_E], Q[_qt], Q[_ql], Q[_qi], [_qr]

    F[1, _qt], F[2, _qt], F[3, _qt] = u * qt, v * qt, w * qt
    F[1, _ql], F[2, _ql], F[3, _ql] = u * ql, v * ql, w * ql
    F[1, _qi], F[2, _qi], F[3, _qi] = u * qi, v * qi, w * qi
    F[1, _qr], F[2, _qr], F[3, _qr] = u * qr, (v + rain_w) * qr, w * qr
    F[1, _E],  F[2, _E],  F[3, _E]  = u *  E, v *  E, w * E
  end
end

# initial condition
const w_max = .6
const Z_max = 1.5
const X_max = 1.5

function single_eddy!(Q, t, x, y, z)
  DFloat = eltype(Q)

  θ_0::DFloat  = 289
  p_0::DFloat  = 101500
  qt_0::DFloat = 7.5 * 1e-3
  z_0::DFloat  = 0

  R_m, cp_m, cv_m, γ = moist_gas_constants(qt_0) 

  # pressure profile assuming hydrostatic and constant θ and qt profiles
  # TODO - check
  # TODO - is z my vertical height?
  p = MSLP * ((p_0 / MSLP)^(R_m / cp_m) -
	       R_m / cp_m * grav / θ_0 / R_m * (y - z_0)
             )^(cp_m / R_m)

  qt::DFloat = qt_0
  ql::DFloat = 0
  qi::DFloat = 0
  qr::DFloat = 0

  T::DFloat = θ_0 * exner(p)
  ρ::DFloat = p / R_m / T

  # TODO should this be more "grid aware"?
  # the velocity is cxalculated as derivative of streamfunction
  U::DFloat = w_max * X_max/Z_max * cos(π * y/Z_max) * cos(2*π * x/X_max)
  W::DFloat = 0
  V::DFloat = 2*w_max * sin(π * y/Z_max) * sin(2*π * x/X_max)

  u = U/ρ
  v = V/ρ
  w = W/ρ

  E = ρ * (grav * y + (1//2)*(u^2 + v^2 + w^2) + internal_energy(T, qt))

  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_qt], Q[_ql], Q[_qi], Q[_qr] = 
            ρ, U, V, W, E, qt, ql, qi, qr

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
                           NumericalFluxes.rosanuv!(x..., eulerflux!,
                                                    wavespeed,
                                                    preflux))

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = single_eddy!(Q, DFloat(0), x...)
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
    outprefix = @sprintf("vtk/single_eddy_%dD_mpirank%04d_step%04d",
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

  brickrange = ntuple(j->range(DFloat(0); length=75+1, stop=1.5), 2)

  topl = BrickTopology(mpicomm, brickrange, periodicity=ntuple(i->true, dim))
  dt = 1e-2 / Ne[1] # not a general purpose dt calculation

  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)
  
end

using Test
let
  timeend = 0.01
  numelem = (75, 75)
  lvls = 3
  dim = 2

  polynomialorder = 4

  for DFloat in (Float64,) #Float32)
    err = zeros(DFloat, lvls)
    for l = 1:lvls
      run(dim, ntuple(j->2^(l-1) * numelem[j], dim),
          polynomialorder, timeend, DFloat)
    end
  end
end

isinteractive() || MPI.Finalize()

nothing
