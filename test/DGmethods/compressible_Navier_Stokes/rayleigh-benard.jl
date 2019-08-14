# Load modules that are used in the CliMA project.
# These are general modules not necessarily specific
# to CliMA
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
using CLIMA.Vtk
using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray 
else
  const ArrayType = Array
end
# For a three dimensional problem 
const _nstate = 6
const _ρ, _U, _V, _W, _E, _QT = 1:_nstate
const stateid = (ρid = _ρ, Uid = _U, Vid = _V, Wid = _W, Eid = _E, QTid = _QT)
const statenames = ("ρ", "U", "V", "W", "E", "QT")

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

const xmin = 0
const ymin = 0
const zmin = 0
const xmax = 2000
const ymax = 400
const zmax = 2000
const C_smag = 0.18
const numelem = (10,2,10)
const polynomialorder = 4

const Δx = (xmax-xmin)/(numelem[1]*polynomialorder+1)
const Δy = (ymax-ymin)/(numelem[2]*polynomialorder+1)
const Δz = (zmax-zmin)/(numelem[3]*polynomialorder+1)
const Δ  = cbrt(Δx * Δy * Δz) 
const dt = 0.005
const timeend = 3000

@inline function diagnostics(Q, aux)
  R_gas::eltype(Q) = R_d
  @inbounds ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
  ρinv = 1 / ρ
  x,y,z = aux[_a_x], aux[_a_y], aux[_a_z]
  u, v, w = ρinv * U, ρinv * V, ρinv * W
  e_int = (E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * grav * z) * ρinv
  q_tot = QT / ρ
  # Establish the current thermodynamic state using the prognostic variables
  TS = PhaseEquil(e_int, q_tot, ρ)
  T = air_temperature(TS)
  P = air_pressure(TS) # Test with dry atmosphere
  (T, P, u, v, w, ρinv)
end

# max eigenvalue
@inline function wavespeed(n, Q, aux, t)
  T, P, u, v, w, ρinv = diagnostics(Q, aux)
  @inbounds abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(T)
end

@inline function cns_flux!(F, Q, VF, aux, t)
  T, P, u, v, w, ρinv = diagnostics(Q, aux)
  @inbounds begin
    ρ, U, V, W, E, QT = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]
    # Inviscid contributions 
    F[1, _ρ], F[2, _ρ], F[3, _ρ]    = U          , V          , W
    F[1, _U], F[2, _U], F[3, _U]    = u * U  + P , v * U      , w * U
    F[1, _V], F[2, _V], F[3, _V]    = u * V      , v * V + P  , w * V
    F[1, _W], F[2, _W], F[3, _W]    = u * W      , v * W      , w * W + P
    F[1, _E], F[2, _E], F[3, _E]    = u * (E + P), v * (E + P), w * (E + P)
    F[1, _QT], F[2, _QT], F[3, _QT] = u * QT  , v * QT     , w * QT 
    
    # Stress tensor
    τ11, τ22, τ33 = ρ * VF[_τ11], ρ * VF[_τ22], ρ * VF[_τ33]
    τ12 = τ21 = ρ * VF[_τ12]
    τ13 = τ31 = ρ * VF[_τ13]
    τ23 = τ32 = ρ * VF[_τ23]
    
    # Viscous contributions
    F[1, _U] += τ11; F[2, _U] += τ12; F[3, _U] += τ13
    F[1, _V] += τ21; F[2, _V] += τ22; F[3, _V] += τ23
    F[1, _W] += τ31; F[2, _W] += τ32; F[3, _W] += τ33
    # Energy dissipation
    vEx, vEy, vEz = ρ * VF[_Ex], ρ * VF[_Ey], ρ*VF[_Ez]
    F[1, _E] += u * τ11 + v * τ12 + w * τ13 + vEx
    F[2, _E] += u * τ21 + v * τ22 + w * τ23 + vEy
    F[3, _E] += u * τ31 + v * τ32 + w * τ33 + vEz
  end
end

# -------------------------------------------------------------------------
# Compute the velocity from the state
const _ngradstates = 4
gradient_vars!(grad_vars, Q, aux, t, _...) = gradient_vars!(grad_vars, Q, aux, t, diagnostics(Q,aux)...)
@inline function gradient_vars!(grad_vars, Q, aux, t, T, P, u, v, w, ρinv)
  @inbounds begin
    ρ, U, V, W, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    ρinv = 1 / ρ
    grad_vars[1], grad_vars[2], grad_vars[3] = ρinv * U, ρinv * V, ρinv * W
    grad_vars[4] = (E * ρinv + R_d * T)
  end
end

# -------------------------------------------------------------------------
# Auxiliary Function
const _nauxstate = 3
const _a_x, _a_y, _a_z, = 1:_nauxstate
@inline function auxiliary_state_initialization!(aux, x, y, z)
  @inbounds begin
    aux[_a_x] = x
    aux[_a_y] = y
    aux[_a_z] = z
  end
end

# -------------------------------------------------------------------------
# Viscous fluxes
const _nviscstates = 11
const _τ11, _τ22, _τ33, _τ12, _τ13, _τ23, _Ex, _Ey, _Ez, _SijSij, _visc = 1:_nviscstates
@inline function compute_stresses!(VF, grad_mat, _...)
  @inbounds begin
    dudx, dudy, dudz = grad_mat[1,1], grad_mat[2,1], grad_mat[3,1] 
    dvdx, dvdy, dvdz = grad_mat[1,2], grad_mat[2,2], grad_mat[3,2] 
    dwdx, dwdy, dwdz = grad_mat[1,3], grad_mat[2,3], grad_mat[3,3] 
    dEdx, dEdy, dEdz = grad_mat[1,4], grad_mat[2,4], grad_mat[3,4] 
    S11, S22, S33 = dudx, dvdy, dwdz
    S12 = (dudy + dvdx) / 2
    S13 = (dudz + dwdx) / 2
    S23 = (dvdz + dwdy) / 2
    SijSij = (S11^2 + S22^2 + S33^2 + 2S12^2 + 2S13^2 + 2S23^2) 
    ν_eddy = sqrt(2*SijSij) * (C_smag * Δ)^2 
    D_eddy = 3ν_eddy
    VF[_τ11] = -2 * ν_eddy * S11
    VF[_τ22] = -2 * ν_eddy * S22
    VF[_τ33] = -2 * ν_eddy * S33
    VF[_τ12] = -2 * ν_eddy * S11
    VF[_τ13] = -2 * ν_eddy * S13
    VF[_τ23] = -2 * ν_eddy * S23
    VF[_Ex], VF[_Ey], VF[_Ez] = -D_eddy * dEdx, -D_eddy * dEdy, -D_eddy * dEdz
    VF[_visc] = ν_eddy
  end
end

# -------------------------------------------------------------------------
# Rayleigh-Benard problem with two fixed walls (prescribed temperatures)
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t)
  @inbounds begin
    x, y, z = auxM[_a_x], auxM[_a_y], auxM[_a_z]
    ρM, UM, VM, WM, EM, QTM = QM[_ρ], QM[_U], QM[_V], QM[_W], QM[_E], QM[_QT]
    UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM
    z   = auxM[_a_z]
    ρP  = ρM
    # Prescribe no-slip wall.
    # Note that with the default resolution this results in an underresolved near-wall layer
    UP  = 0 
    VP  = 0 
    WP  = 0 
    # Weak Boundary Condition Imposition
    # In the limit of Δ → 0, the exact boundary values are recovered at the "M" or minus side. 
    # The weak enforcement of plus side states ensures that the boundary fluxes are consistently calculated.
    if auxM[_a_z] < 0.001
      E_intP = ρP * cv_d * (T_bot - T_0)
    else
      E_intP = ρP * cv_d * (T_top - T_0) 
    end
    QP[_ρ] = ρP
    QP[_U] = UP
    QP[_V] = VP
    QP[_W] = WP
    QP[_E] = (E_intP + (UP^2 + VP^2 + WP^2)/(2*ρP) + ρP * grav * z)
    VFP .= VFM
    VFP[_τ33] = 0 
    VFP[_Ez] = 0 
    nothing
  end
end

@inline stresses_boundary_penalty!(VF, _...) = VF.=0

@inline function stresses_penalty!(VF, nM, grad_varsM, QM, aM, grad_varsP, QP, aP, t)
  @inbounds begin
    n_Δgrad_vars = similar(VF, Size(3, _ngradstates))
    for j = 1:_ngradstates, i = 1:3
      n_Δgrad_vars[i, j] = nM[i] * (grad_varsP[j] - grad_varsM[j]) / 2
    end
    compute_stresses!(VF, n_Δgrad_vars)
  end
end

"""
The function source! collects all the individual source terms 
associated with a given problem. We do not define sources here, 
rather we only call those source terms which are necessary based
on the governing equations. 
by terms defined elsewhere
"""
@inline function source!(S,Q,aux,t)
  S .= 0
  @inbounds begin
    source_geopot!(S, Q, aux, t)
  end
end

"""
Geopotential source term. Gravity forcing applied to the vertical
momentum equation
"""
@inline function source_geopot!(S,Q,aux,t)
  @inbounds begin
    ρ, U, V, W, E  = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]
    S[_W] -= ρ * grav
  end
end

"""
Initial conditions for the Rayleigh-Benard problem with two fixed walls 
"""
const T_bot     = 320
const T_lapse   = -0.04
const T_top     = T_bot + T_lapse*zmax
const α_thermal = 0.0034

using Random
const seed = MersenneTwister(0)
function rayleigh_benard!(dim, Q, t, x, y, z, _...)
  DFloat                = eltype(Q)
  γ::DFloat             = 7 // 5
  R_gas::DFloat         = R_d
  c_p::DFloat           = cp_d
  c_v::DFloat           = cv_d
  p0::DFloat            = MSLP
  q_tot::DFloat         = 0
  δT::DFloat            = z != 0 ? rand(seed, DFloat)/100 : 0 
  δw::DFloat            = z != 0 ? rand(seed, DFloat)/100 : 0
  ΔT                    = T_lapse * z + δT
  T                     = T_bot + ΔT 
  P                     = p0*(T/T_bot)^(grav/R_gas/T_lapse)
  ρ                     = P / (R_gas * T)
  U, V, W               = 0.0 , 0.0 , ρ * δw
  E_int                 = ρ * c_v * (T-T_0)
  E_pot                 = ρ * grav * z
  E_kin                 = ρ * 0.5 * δw^2 
  E                     = E_int + E_pot + E_kin
  @inbounds Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E], Q[_QT]= ρ, U, V, W, E, ρ * q_tot
end

function run(mpicomm, dim, Ne, N, timeend, DFloat, dt)

  brickrange = (range(DFloat(xmin), length=Ne[1]+1, DFloat(xmax)),
                range(DFloat(ymin), length=Ne[2]+1, DFloat(ymax)),
                range(DFloat(zmin), length=Ne[3]+1, DFloat(zmax)))
  
  # User defined periodicity in the topl assignment
  # brickrange defines the domain extents
  topl = BrickTopology(mpicomm, brickrange, periodicity=(true,true,false))

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N)
  
  numflux!(x...) = NumericalFluxes.rusanov!(x..., cns_flux!, wavespeed)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., cns_flux!, bcstate!, wavespeed)

  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = cns_flux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!, 
                           number_gradient_states = _ngradstates,
                           number_viscous_states = _nviscstates,
                           gradient_transform! = gradient_vars!,
                           viscous_transform! = compute_stresses!,
                           viscous_penalty! = stresses_penalty!,
                           viscous_boundary_penalty! = stresses_boundary_penalty!,
                           auxiliary_state_length = _nauxstate,
                           auxiliary_state_initialization! =
                           auxiliary_state_initialization!,
                           source! = source!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = rayleigh_benard!(Val(dim), Q, 0, x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  lsrk = LSRK54CarpenterKennedy(spacedisc, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(5, mpicomm) do (s=false)
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

#  npoststates = 5
#  _o_T, _o_dEdz, _o_u, _o_v, _o_w = 1:npoststates
#  postnames =("T", "dTdz", "u", "v", "w")
#  postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

  step = [0]
  cbvtk = GenericCallbacks.EveryXSimulationSteps(10000) do (init=false)
#    DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc, Q) do R, Q, QV, aux
#      @inbounds let
#        (T, P, u, v, w, _)= diagnostics(Q, aux)
#        R[_o_dEdz] = QV[_Ez]
#        R[_o_u] = u
#        R[_o_v] = v
#        R[_o_w] = w
#        R[_o_T] = T
#      end
#    end
    mkpath("./vtk-rb-bc/")
    outprefix = @sprintf("./vtk-rb-bc/rb_%dD_mpirank%04d_step%04d", dim,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, spacedisc, statenames)
    
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  # Print some end of the simulation information
  engf = norm(Q)
  @info @sprintf """Finished
  norm(Q)            = %.16e
  norm(Q) / norm(Q₀) = %.16e
  norm(Q) - norm(Q₀) = %.16e""" engf engf/eng0 engf-eng0
  engf/eng0
end

using Test
let
  MPI.Initialized() || MPI.Init()
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

  for DFloat in (Float64,) 
    for dim = 3:3
      engf_eng0 = run(mpicomm, dim, numelem[1:dim], polynomialorder, timeend,
                      DFloat, dt)
      @test engf_eng0 ≈ 1.0000734828671902e+00
    end
  end
end
nothing
